import streamlit as st
import requests
import os
import threading
from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class SharedState:
    def __init__(self):
        self.parsed_data = {}
        self.corpus = []
        self.corpus_embeddings = None
        self.is_loaded = False

state = SharedState()

def load_data():
    """Load and prepare data before starting Flask"""
    logger.info("Starting data loading process...")
    
    for root, _, files in os.walk('parsed_data'):
        logger.info(f"Processing directory: {root}")
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  
                            if 'repo_' in file:
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

    user_input = request.json.get('input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
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
        sources = []
        
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
Provide a clear and specific answer, citing the relevant documentation where possible. If the documentation you feel is incomplete you can answer based on your knowledge DO NOT say documentation doesn't say this or similar."""

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

st.title("Sugar Labs Chatbot")
st.write("Ask a question about contributing to Sugar Labs:")

user_input = st.text_area("Your question:")

if st.button("Submit"):
    if user_input.strip():
        api_url = os.getenv("API_URL", "http://localhost:5000/api/chatbot")
        try:
            response = requests.post(api_url, json={"input": user_input})
            if response.status_code == 200:
                data = response.json()
                st.write("Chatbot response:")
                st.write(data['response'])
                
                if data['sources']:
                    st.write("\nSources used:")
                    for source in data['sources']:
                        st.write(f"- {source}")
                
                with st.expander("Debug Information"):
                    st.write("Documents loaded:", data['debug']['docs_loaded'])
                    st.write("Matches found:", data['debug']['matches_found'])
                    st.write("Similarity threshold:", data['debug']['threshold'])
                    
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
        st.warning("Please enter a question.")