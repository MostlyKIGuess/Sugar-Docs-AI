import streamlit as st
import requests
import os
import threading
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import logging
import base64
import re
from PIL import Image
import io
# Add to imports
from matrix_client.client import MatrixClient
from matrix_client.api import MatrixRequestError
import asyncio
import time
from time import perf_counter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class SharedState:
    def __init__(self):
        self.parsed_data = {}
        self.corpus = []
        self.corpus_embeddings = None
        self.is_loaded = False

state = SharedState()


class PerformanceMonitor:
    def __init__(self):
        self.queries = {}
        self._lock = threading.Lock()

    def start_query(self, query_id):
        with self._lock:
            self.queries[query_id] = perf_counter()

    def end_query(self, query_id):
        with self._lock:
            if query_id in self.queries:
                duration = perf_counter() - self.queries.pop(query_id)
                logger.info(f"Query {query_id} took {duration:.2f} seconds")

class MatrixBot:
    def __init__(self, state, api_url):
        self.state = state
        self.perf_monitor = PerformanceMonitor()
        self.api_url = api_url
        self.client = MatrixClient(os.getenv('MATRIX_HOMESERVER', 'https://matrix.org'))
        self.api = self.client.api
        self.messages = []
        self.connected = False
        self.retry_count = 0
        self.max_retries = 3
        self.is_thinking = False
        self._request_lock = threading.RLock()
        self._active_requests = {}  # Track requests per user
        self._request_timeouts = {}
        self.request_timeout = 30  # seconds
        self._image_context = {}  # images per user
        self._image_timeout = 300 
        self.homeserver = os.getenv('MATRIX_HOMESERVER', 'https://matrix.org')
        self._global_lock = threading.Lock()
        self.lock = threading.Lock()
        self.request_in_progress = False
        self._message_queue = []
        self._worker = None
        self._should_run = True
        self._last_sync = 0
        self._sync_delay = 30  # seconds
        self.sync_token = None
        self._last_message_time = 0
        self._listener_thread = None
        self._media_base_url = f"{self.homeserver}/_matrix/client/v1"
        self._max_image_size = 10 * 1024 * 1024 
        self._latest_image = None
        self._latest_image_description = None

        
    def _convert_mxc_url(self, mxc_url):
        """Convert MXC URL to HTTP URL using v1 client API"""
        if not mxc_url.startswith('mxc://'):
            return None
            
        parts = mxc_url.split('/')
        if len(parts) != 4:
            return None
            
        server_name = parts[2]
        media_id = parts[3]
        
        return f"{self._media_base_url}/media/download/{server_name}/{media_id}"


    def get_image_description(self, image_data):
        """Call Gemini API to get image description"""
        try:
            # image_bytes = base64.b64encode(image_data).decode('utf-8')
            image_prompt = "Describe this image in detail, focusing on any musical or programming elements visible in MusicBlocks."
            response = model.generate_content([image_data, image_prompt])
            return response.text
        except Exception as e:
            logger.error(f"Error getting image description: {str(e)}")
            return "No description available"
        
    def handle_image(self, room, event):
        """Handle image upload with v1 client API"""
        if 'url' not in event['content']:
            return
                
        sender = event['sender']
        mxc_url = event['content']['url']
        
        try:
            http_url = self._convert_mxc_url(mxc_url)
            if not http_url:
                logger.error(f"Invalid MXC URL: {mxc_url}")
                return
                    
            # Try thumbnail first
            if 'info' in event['content'] and 'thumbnail_url' in event['content']['info']:
                thumb_mxc = event['content']['info']['thumbnail_url']
                thumb_url = self._convert_mxc_url(thumb_mxc)
                if thumb_url:
                    http_url = thumb_url
            
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Accept': 'image/*'
            }
            
            logger.info(f"Downloading image from: {http_url}")
            response = requests.get(
                http_url,
                headers=headers,
                timeout=10,  # Set a timeout for the request
                stream=True
            )
            
            if response.status_code == 200:
                logger.info("Image downloaded successfully")
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    logger.error(f"Invalid content type: {content_type}")
                    return

                content_length = int(response.headers.get('Content-Length', 0))
                if content_length > self._max_image_size:
                    logger.error(f"Image too large: {content_length} bytes")
                    return

                image_data = response.content
                response.close()
                # Validate image
                try:
                    logger.info("Processing image")
                    image = Image.open(io.BytesIO(image_data))

                    # encoded_data = base64.b64encode(final_data).decode('utf-8')
                    logger.info("Getting Description " )

                    # Call Gemini API for image information
                    image_description = self.get_image_description(image)
                    logger.info(f"Image description: {image_description}")

                    self._latest_image_description = image_description
                    logger.info("Description added")
                    return
                except Exception as e:
                    logger.error(f"Invalid image data: {e}")
                    return
               
                    
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                
                with self._global_lock:
                    self._image_context[sender] = {
                        'data': encoded_data,
                        'content_type': content_type,
                        'timestamp': time.time(),
                        'event_id': event['event_id']
                    }
                logger.info(f"Successfully stored image from {sender}")
            else:
                logger.error(f"Failed to download image: {response.status_code} - {response.text}")
                    
        except requests.exceptions.Timeout:
            logger.error("Image download timed out")
        except Exception as e:
            logger.error(f"Error handling image: {str(e)}")

    def message_handler(self, room, event):
        """Handle both text and image messages with proper locking"""
        logger.info(f"Received Matrix event: {event}")
        
        if event['type'] == "m.room.message":
            msgtype = event['content']['msgtype']
            sender = event['sender']
            
            # Handle images
            if msgtype == "m.image":
                logger.info(f"Processing image from {sender}")
                try:
                    with self._global_lock:
                        self.room.send_text("🖼️ Image received. Processing...")
                        self.handle_image(room, event)
                        # currently using latest image
                        # if self._latest_image:
                        #     self.room.send_text("✓ Image received. You can now use !ask with your question.")
                        # if sender in self._image_context:
                        if self._latest_image_description:
                            self.room.send_text("✓ Image received. You can now use !ask with your question.")
                        #     self.room.send_text("✓ Image received. You can now use !ask with your question.")
                except Exception as e:
                    logger.error(f"Error handling image: {e}")
                    self.room.send_text("Sorry, there was an error processing the image.")
            
            # Handle text messages    
            if msgtype == "m.text":
                body = event['content']['body']
                logger.info(f"Matrix message from {sender}: {body}")
                
                if body.startswith('!ask'):
                    query = body[5:].strip()
                    with self._request_lock:
                        if self.request_in_progress:
                            self.room.send_text("Another request is being processed. Please wait.")
                            return
                        self.request_in_progress = True
                    
                    try:
                        self.is_thinking = True
                        self.room.send_text("🤔 Thinking...")
                        
                        # Build request data
                        request_data = {"input": query}
                        
                        # Add image if available
                        # with self._global_lock:
                            # if sender in self._image_context:
                            #     request_data["image"] = self._image_context[sender]["data"]
                            # if self._latest_image:
                                # request_data["image"] = self._latest_image["data"]
                        # with self._global_lock:
                        #     if self._latest_image:
                        #         request_data["description"] = self._latest_image["description"]
                        
                        if self._latest_image_description:
                            request_data['input'] =  f"{query} + Latest Image Description: {self._latest_image_description}"
                            self._latest_image_description = None
                        # Make API call
                        response = requests.post(self.api_url, json=request_data)
                        
                        if response.status_code == 200:
                            data = response.json()
                            self.room.send_text(data['response'])
                            self.messages.append({
                                'query': query,
                                'response': data['response'],
                                'sources': data.get('sources', []),
                                'timestamp': time.strftime('%H:%M:%S'),
                                'debug': data.get('debug', {})
                            })
                        else:
                            logger.error(f"API error: {response.status_code}")
                            self.room.send_text("Sorry, there was an error processing your request.")
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        self.room.send_text("Sorry, there was an error processing your request.")
                    finally:
                        with self._request_lock:
                            self.request_in_progress = False
                        self.is_thinking = False
                        
    def connect(self):
        """
        Connect and join the specified Matrix room with retries. Log all steps.
        """
        while self.retry_count < self.max_retries:
            try:
                logger.info(f"Login attempt {self.retry_count + 1} as {os.getenv('MATRIX_USERNAME')}")
                
                # Clean up username
                username = os.getenv('MATRIX_USERNAME')
                if '@' in username:
                    username = username.split(':')[0].replace('@', '')
                
                self.token = self.client.login(
                    username=username,
                    password=os.getenv('MATRIX_PASSWORD')
                )
                logger.info("Login successful")
                
                room_id = os.getenv('MATRIX_ROOM_ID')
                logger.info(f"Attempting to join room {room_id}")
                
                try:
                    self.room = self.client.join_room(room_id)
                    logger.info(f"Joined room {room_id}")
                    self.room.add_listener(self.message_handler)
                    self.connected = True
                    logger.info("Matrix bot ready")
                    
                    # Start listener thread
                    threading.Thread(target=self.client.listen_forever, daemon=True).start()
                    return True
                    
                except MatrixRequestError as e:
                    logger.error(f"Room join error: {e}, code={e.code}")
                    if e.code == 403:
                        logger.error("Need room invite first")
                        return False
                    raise
                    
            except MatrixRequestError as e:
                logger.error(f"Matrix login error: {str(e)}")
                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return False
                
        logger.error("Max retries reached - Matrix connection failed")
        return False



class Section:
    def __init__(self, title, content, number):
        self.title = title
        self.content = content
        self.number = number
        self.subsections = []

class GuideParser:
    def __init__(self):
        self.sections = {}
        self.section_titles = []
        
    def parse_guide(self, content):
        """Parse guide content into hierarchical sections"""
        current_section = None
        section_content = []
        
        for line in content.split('\n'):
            if re.match(r'^\d+\.\d+(\.\d+)?\s+', line):  # Section number found
                if current_section:
                    self.sections[current_section.number] = current_section
                
                number = line.split()[0]
                title = ' '.join(line.split()[1:])
                current_section = Section(title, '', number)
                self.section_titles.append(f"{number} {title}")
            else:
                if current_section:
                    current_section.content += line + '\n'
                    
        return self.sections
    
    
GUIDE_FILES = ['guidemusicblocks.txt', 'usingmusicblocks.txt']

def load_data():
    """Load and prepare data before starting Flask"""
    logger.info("Starting data loading process...")
    
    guide_parser = GuideParser()
    
    guide_path = os.path.join('parsed_data', 'guidemusicblocks.txt')
    if os.path.exists(guide_path):
        with open(guide_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            state.guide_sections = guide_parser.parse_guide(content)
            state.section_titles = guide_parser.section_titles
            
    guide_files = ['guidemusicblocks.txt', 'usingmusicblocks.txt']

    for root, _, files in os.walk('parsed_data'):
        logger.info(f"Processing directory: {root}")
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  
                            if file in guide_files:
                                content = f"GUIDE DOCUMENTATION: {content}"
                            elif 'repo_' in file:
                                content = f"Repository Information: {content}"
                            state.parsed_data[file] = content
                            state.corpus.append(content)
                            logger.info(f"Loaded file: {file} (length: {len(content)})")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
    if state.corpus:
        try:
            logger.info(f"Generating embeddings for {len(state.corpus)} documents...")
            state.corpus_embeddings = embedder.encode(state.corpus, convert_to_tensor=True)
            logger.info(f"Embeddings generated with shape: {state.corpus_embeddings.shape}")
            state.is_loaded = True
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
    else:
        logger.error("No documents found in corpus!")

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not state.is_loaded:
        return jsonify({'error': 'Document corpus not loaded'}), 500

    user_input = request.json.get('input', '')
    image_data = request.json.get('image')
    
    try:
        sources = []
        if image_data:
            # Process image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            image_prompt = "Describe this image in detail, focusing on any musical or programming elements visible in MusicBlocks."
            image_description = model.generate_content([image, image_prompt]).text
            
            # Create guide files mapping with focus on MusicBlocks sections
            musicblocks_files = ['guidemusicblocks.txt', 'usingmusicblocks.txt']
            corpus_mapping = {file: idx for idx, file in enumerate(state.parsed_data.keys())}
            musicblocks_indices = [idx for file, idx in corpus_mapping.items() 
                                 if any(mb_file in file for mb_file in musicblocks_files)]
            
            # Create search query combining user input and image description
            query = f"MusicBlocks Documentation: {user_input}\n{image_description}"
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            
            # Search specifically in MusicBlocks sections with higher threshold
            musicblocks_embeddings = state.corpus_embeddings[musicblocks_indices]
            section_hits = util.semantic_search(query_embedding, musicblocks_embeddings, top_k=3)
            
            # Map hits back to original indices and boost MusicBlocks section scores
            mapped_hits = [{
                'corpus_id': musicblocks_indices[hit['corpus_id']],
                'score': hit['score'] * 1.2  # Boost MusicBlocks section relevance
            } for hit in section_hits[0]]
            
            # Search full corpus with lower threshold
            all_hits = util.semantic_search(query_embedding, state.corpus_embeddings, top_k=5)
            
            # Combine results preserving original indices
            hits = mapped_hits + [
                hit for hit in all_hits[0] 
                if hit['corpus_id'] not in [h['corpus_id'] for h in mapped_hits]
            ]
            
            all_matches = []
            guide_threshold = 0.25
            general_threshold = 0.2
            relevant_info = ""
            for hit in hits:
                source = list(state.parsed_data.keys())[hit['corpus_id']]
                is_guide = any(guide in source for guide in GUIDE_FILES)
                threshold = guide_threshold if is_guide else general_threshold
                
                if hit['score'] > threshold:
                    content_preview = state.corpus[hit['corpus_id']][:100]
                    all_matches.append({
                        'source': source,
                        'score': hit['score'],
                        'preview': content_preview
                    })
                    sources.append(source)
                    relevant_info += f"\n\nFrom {source} (relevance: {hit['score']:.2f}):\n{state.corpus[hit['corpus_id']][:2000]}"

            prompt = f"""You are a musicblocks teacher and knows a lot about musicblocks and music theory, users will ask you questions based on their images pleasae understand them and help and give Music Theory concepts. Please help the student based on the user input, image might not have what they are asking so guide them using documentation and tell them to search using search widget on the site. But students will ask you questions based on the image as well and you have to help them.
User question: {user_input}
Image description: {image_description}
Documentation: {relevant_info}. If you didn't find anything relevant in documentation then you can use your own knowledge to answer the question."""

            response = model.generate_content(prompt)
            
            return jsonify({
                'response': response.text,
                'sources': sources,
                'image_description': image_description,
                'debug': {
                    'docs_loaded': len(state.corpus),
                    'matches_found': len(sources),
                    'threshold': threshold,
                    'all_matches': all_matches
                }
            })
        else:
            # Process text input with embeddings
            query_embedding = embedder.encode(user_input, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, state.corpus_embeddings, top_k=10)
            
            logger.info("All matches found:")
            all_matches = []
            for hit in hits[0]:
                source = list(state.parsed_data.keys())[hit['corpus_id']]
                score = hit['score']
                content_preview = state.corpus[hit['corpus_id']][:100]  
                all_matches.append({
                    'source': source,
                    'score': score,
                    'preview': content_preview
                })
                logger.info(f"Score: {score:.4f} | Source: {source} | Preview: {content_preview}...")

            threshold = 0.2  
            relevant_info = ""
            
            for hit in hits[0]:
                if hit['score'] > threshold:
                    source = list(state.parsed_data.keys())[hit['corpus_id']]
                    sources.append(source)
                    doc_content = state.corpus[hit['corpus_id']][:1000]  
                    relevant_info += f"\n\nFrom {source} (relevance: {hit['score']:.2f}):\n{doc_content}"

            prompt = f"""You are a Sugar Labs assistant. Answer the user's question based on the following documentation sections.
If multiple sources provide relevant information, combine them in your response.
Remember Sugar uses AGPLv3 license, so all contributions must be compatible with this license.
Documentation sections:
{relevant_info}
Question: {user_input}
Provide a clear and specific answer, citing the relevant documentation where possible."""

            response = model.generate_content(prompt)
        
            return jsonify({
                'response': response.text,
                'sources': sources,
                'debug': {
                    'docs_loaded': len(state.corpus),
                    'matches_found': len(sources),
                    'threshold': threshold,
                    'all_matches': all_matches  
                }
            })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

logger.info("Initializing data...")
load_data()

def run_flask():
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)

threading.Thread(target=run_flask).start()



if "request_in_progress" not in st.session_state:
    st.session_state.request_in_progress = False
    

st.title("Sugar Labs Chatbot")

tab1, tab2 = st.tabs(["Direct Chat", "Matrix Channel"])

with tab1:
    st.write("Ask a question about contributing to Sugar Labs:")

user_input = st.text_area("Your question:")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "heic", "heif"])

if st.button("Submit"):
    if user_input.strip() or uploaded_file:
        api_url = os.getenv("API_URL", "http://localhost:5000/api/chatbot")
        data = {"input": user_input}
        
        if uploaded_file:
            image_bytes = uploaded_file.read()
            data["image"] = base64.b64encode(image_bytes).decode('utf-8')
        
        try:
            response = requests.post(api_url, json=data)
            if response.status_code == 200:
                data = response.json()
                st.write("# Chatbot response:")
                st.write(data['response'])
                
                if data.get('image_description'):
                    st.write("\n# Image description:")
                    st.write(data['image_description'])
                    
                if data['sources']:
                    st.write("\n**Sources used:**")
                    for source in data['sources']:
                        st.write(f"- {source}")
                
                with st.expander("Debug Information"):
                    st.write("Documents loaded:", data['debug'].get('docs_loaded', 'N/A'))
                    st.write("Matches found:", data['debug'].get('matches_found', 'N/A'))
                    st.write("Similarity threshold:", data['debug'].get('threshold', 'N/A'))
                    
                    if 'all_matches' in data['debug']:
                        st.write("\nAll matches (including below threshold):")
                        for match in data['debug']['all_matches']:
                            st.write(f"\nSource: {match['source']}")
                            st.write(f"Score: {match['score']:.4f}")
                            st.write("Preview:", match['preview'])
            else:
                st.error(f"Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Error: Unable to connect to the API.")
    else:
        st.warning("Please enter a question or upload an image.")
    
with tab2:
    st.write("Matrix Channel Messages")
    if 'matrix_bot' not in st.session_state:
        st.info("Initializing Matrix bot...")
        st.session_state.matrix_bot = MatrixBot(state, os.getenv("API_URL", "http://localhost:5000/api/chatbot"))
        threading.Thread(target=st.session_state.matrix_bot.connect).start()
    
    if st.session_state.matrix_bot.connected:
        st.success("Connected to Matrix room")
    else:
        st.error("Not connected to Matrix room. Please check credentials and room invite.")
        
    # Display Matrix messages 
    for msg in st.session_state.matrix_bot.messages:
        st.write(f"**Question:** {msg['query']}")
        st.write(f"**Answer:** {msg['response']}")
        if msg['sources']:
            st.write("**Sources:**")
            for source in msg['sources']:
                st.write(f"- {source}")
        with st.expander("Debug Info"):
            st.write(msg['debug'])
    
    # Auto-refresh
    if st.button("Refresh Messages"):
        st.rerun()