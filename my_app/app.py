# Flask and API-related imports
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import requests
import subprocess

# IBM Watson imports
from ibm_watson import AssistantV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# LangChain and embedding imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Data processing and utility imports
import os
import uuid
import time
import re
import hashlib
import pickle
import shutil
import json
import numpy as np
from typing import List, Tuple, Optional, Union
from datetime import datetime

# Matplotlib and visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# Text and NLP-related imports
import textwrap
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# File handling imports
import io
import base64
import PyPDF2

# Image processing imports
from PIL import Image, ImageDraw, ImageFont


# Directory for saving LLM outputs
DEBUG_OUTPUT_DIR = './debug_outputs'
if not os.path.exists(DEBUG_OUTPUT_DIR):
    os.makedirs(DEBUG_OUTPUT_DIR)

# Directory for LLM interaction logs
LLM_LOGS_DIR = './llm_logs'
if not os.path.exists(LLM_LOGS_DIR):
    os.makedirs(LLM_LOGS_DIR)

# Create Flask app with correct template and static folders
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Simplified CORS configuration
CORS(app)

# Directory configurations
UPLOAD_FOLDER = 'upload'
CHROMA_DB_DIR = './chroma_db'
CACHE_DIR = './embedding_cache'

# Create necessary directories
for directory in [UPLOAD_FOLDER, CHROMA_DB_DIR, CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Watson Assistant credentials and configuration
API_KEY = "KXxe28FCGlNkQOMaZeORNVFilSSUYBF7sYBIvnYM0Xp7"
SERVICE_URL = "https://api.eu-gb.assistant.watson.cloud.ibm.com/instances/a67b8127-fcef-4389-acf3-5763e81017f8"
WORKSPACE_ID = "c68e1855-d988-4f7d-af08-8f5e9a0d2c47"

# Model and processing configurations
LLM_MODEL = "granite3.2:8b"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
reserved_cores = 2
MAX_WORKERS = max(1, os.cpu_count() - reserved_cores)

# MMR retrieval configurations
USE_MMR = True  # Set to True to use MMR, False to use standard similarity search
MMR_K = 8  # Number of documents to retrieve after MMR reranking
MMR_LAMBDA = 0.7  # Balance between relevance (1) and diversity (0)

def log_llm_interaction(prompt, response, interaction_type="general"):
    try:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique filename based on the interaction type
        output_filename = f"{LLM_LOGS_DIR}/llm_{interaction_type}_{timestamp}.json"
        
        # Create a structured log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "prompt": prompt,
            "response": response,
            "model": LLM_MODEL
        }
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        print(f"📋 LLM interaction logged to: {output_filename}")
        return True
    except Exception as e:
        print(f"Warning: Could not log LLM interaction: {e}")
        return False

def clear_chroma_db():
    """Clear the existing Chroma database to solve dimension mismatch issues"""
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Clearing existing Chroma database at {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)
        os.makedirs(CHROMA_DB_DIR)
        print("Chroma database cleared successfully")
    else:
        print(f"No existing Chroma database found at {CHROMA_DB_DIR}")

def get_cache_key(content: Union[str, bytes]) -> str:
    """Generate a cache key for content"""
    if isinstance(content, str):
        content_bytes = content.encode()
    else:
        content_bytes = content
    return hashlib.md5(content_bytes).hexdigest()

def process_batch(batch: List[str], embeddings: OllamaEmbeddings) -> List[np.ndarray]:
    """Process a batch of texts to get their embeddings"""
    return embeddings.embed_documents(batch)

def get_simple_embeddings(texts):
    """Simple fallback embedding method when Ollama fails"""
    
    def hash_text(text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 8)
    
    # Create simple pseudo-embeddings based on text hashing
    # This is just a fallback and won't provide good semantic search
    embeddings = []
    for text in texts:
        # Generate a deterministic but simple vector from text
        seed = hash_text(text)
        np.random.seed(seed)
        embeddings.append(np.random.rand(384))  # 384-dim vector
    
    return embeddings

def ensure_ollama_running():
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code != 200:
            raise Exception("Ollama not responding properly")
    except Exception:
        # Try to start Ollama
        print("Starting Ollama...")
        try:
            subprocess.Popen(["ollama", "serve"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
            # Wait for Ollama to start
            for _ in range(10):
                time.sleep(2)
                try:
                    response = requests.get("http://localhost:11434/api/version", timeout=2)
                    if response.status_code == 200:
                        print("Ollama started successfully")
                        # Pull the models we need
                        subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=True)
                        subprocess.run(["ollama", "pull", LLM_MODEL], check=True)
                        return True
                except:
                    pass
            raise Exception("Failed to start Ollama")
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False
    return True

def create_embeddings():
    """Create embeddings with proper configuration for current version"""
    try:
        # First try the simplest configuration
        return OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"Basic embedding configuration failed: {e}")
        try:
            # Try with explicit base_url
            return OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url="http://localhost:11434"
            )
        except Exception as e:
            print(f"Embedding with base_url failed: {e}")
            # Last resort - use a completely different embedding model
            try:
                return HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
            except Exception as e:
                print(f"Fallback to HuggingFace failed: {e}")
                return None

def process_pdf(file_path: str) -> Tuple[Optional[RecursiveCharacterTextSplitter], 
                                        Optional[Chroma], 
                                        Optional[any]]:
    """Process PDF with optimized embedding for larger files"""
    print(f"Processing PDF at {file_path}")
    
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Adjust parameters based on file size
        is_large_file = file_size > 5 * 1024 * 1024  # 5MB threshold
        is_very_large_file = file_size > 10 * 1024 * 1024  # 10MB threshold
        
        # Load PDF
        loader = PyMuPDFLoader(file_path)
        data = loader.load()
        print(f"PDF loaded with {len(data)} pages")

        # Adjust chunk settings for larger files
        chunk_size = CHUNK_SIZE
        chunk_overlap = CHUNK_OVERLAP
        
        if is_large_file:
            # For large files, increase chunk size to reduce number of chunks
            chunk_size = int(CHUNK_SIZE * 1.5)
            chunk_overlap = int(CHUNK_OVERLAP * 0.7)  # Reduce overlap
            print(f"Large file detected. Using adjusted chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        if is_very_large_file:
            # For very large files, use even larger chunks
            chunk_size = CHUNK_SIZE * 2
            chunk_overlap = int(CHUNK_OVERLAP * 0.5)  # Further reduce overlap
            print(f"Very large file detected. Using large chunk size: {chunk_size}, overlap: {chunk_overlap}")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(data)
        chunk_count = len(chunks)
        print(f"Split into {chunk_count} chunks")

        # Try embedding with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create embeddings with proper configuration
                embeddings = create_embeddings()
                if embeddings is None:
                    raise Exception("Failed to create any embedding model")
                
                # Process in smaller batches to avoid memory issues
                # Adjust batch size based on file size
                if chunk_count < 100:
                    initial_batch_size = 50
                    subsequent_batch_size = 20
                else:
                    initial_batch_size = max(30, int(chunk_count * 0.1))
                    subsequent_batch_size = max(10, int(chunk_count * 0.05))
                print(f"Batch sizes set: initial={initial_batch_size}, subsequent={subsequent_batch_size}")

                vectorstore = Chroma.from_documents(
                    documents=chunks[:min(len(chunks), initial_batch_size)],
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_DIR
                )
                
                # If we got here, embedding worked, so add remaining chunks if any
                if len(chunks) > initial_batch_size:
                    total_batches = (len(chunks) - initial_batch_size - 1) // subsequent_batch_size + 1
                    print(f"Adding remaining chunks in {total_batches} batches")
                    
                    for i in range(initial_batch_size, len(chunks), subsequent_batch_size):
                        batch = chunks[i:i+subsequent_batch_size]
                        if batch:
                            vectorstore.add_documents(batch)
                            current_batch = (i - initial_batch_size) // subsequent_batch_size + 1
                            print(f"Added batch {current_batch}/{total_batches}")
                
                # Dynamically set MMR_FETCH_K to the actual number of chunks
                if chunk_count < 50:
                    mmr_fetch_k = chunk_count
                else:
                    # Use 30% of the total chunks with a minimum value of 50
                    mmr_fetch_k = min(chunk_count, max(50, int(chunk_count * 0.3)))
                print(f"Dynamically set mmr_fetch_k to {mmr_fetch_k}")
                
                # Create retriever with MMR if enabled
                if USE_MMR:
                    print(f"Using MMR retrieval (k={MMR_K}, fetch_k={mmr_fetch_k}, lambda={MMR_LAMBDA})")
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": MMR_K,
                            "fetch_k": mmr_fetch_k,  # Use dynamic value based on chunk count
                            "lambda_mult": MMR_LAMBDA
                        }
                    )
                else:
                    # Standard similarity search
                    print("Using standard similarity search retrieval")
                    retriever = vectorstore.as_retriever()
                
                print("PDF processing completed successfully")
                return (text_splitter, vectorstore, retriever)
                
            except Exception as e:
                print(f"Embedding error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # Fallback to direct text on final attempt
                    print("Falling back to direct text processing")
                    # For very large files, limit the text size to avoid memory issues
                    if is_very_large_file:
                        print("Very large file - using subset of text")
                        sample_pages = data[:5] + data[len(data)//2:len(data)//2+5] + data[-5:]
                        full_text = "\n\n".join([doc.page_content for doc in sample_pages])
                    else:
                        full_text = "\n\n".join([doc.page_content for doc in data])
                    return None, None, full_text
                time.sleep(2)  # Wait before retry

    except Exception as e:
        print(f"PDF processing error: {e}")
        return None, None, None

def get_conversation_text():
    print("Starting conversation retrieval...")
    authenticator = IAMAuthenticator(API_KEY)
    assistant = AssistantV1(
        version='2021-11-27',
        authenticator=authenticator
    )
    assistant.set_service_url(SERVICE_URL)
    
    # List of phrases to filter out from user messages
    filter_phrases = [
        "Start Analysis", 
        "Perform More Analysis", 
        "More Analysis", 
        "Additional Analysis",
        "Generate Analysis",
        "Continue Analysis",
        "Analyze Document",
        "Run Analysis",
        "Do Analysis"
    ]
    
    analysis_phrase = "Start Analysis"
    analysis_found = False
    latest_conversation_id = None
    collected_logs = []

    try:
        max_attempts = 10
        attempts = 0
        while not analysis_found and attempts < max_attempts:
            print(f"Polling Watson logs, attempt {attempts+1}...")
            newest_log_response = assistant.list_logs(
                workspace_id=WORKSPACE_ID,
                sort='-request_timestamp',
                page_limit=100
            ).get_result()
            logs = newest_log_response.get('logs', [])
            if not logs:
                print("No logs found for this workspace yet.")
            else:
                current_conversation_id = logs[0]['request']['context'].get('conversation_id')
                if latest_conversation_id is None:
                    latest_conversation_id = current_conversation_id

                latest_conversation_logs = [
                    log for log in logs 
                    if log['request']['context'].get('conversation_id') == latest_conversation_id
                ]
                latest_conversation_logs.sort(key=lambda log: log['request_timestamp'])

                for log_item in latest_conversation_logs:
                    user_text = log_item["request"]["input"].get("text", "")
                    if analysis_phrase in user_text:
                        analysis_found = True
                        collected_logs = latest_conversation_logs
                        print("Found analysis phrase in conversation logs.")
                        break

            if not analysis_found:
                print("Analysis phrase not found. Waiting 5 seconds...")
                attempts += 1
                time.sleep(5)
        
        if not analysis_found:
            print("Analysis phrase not found after maximum attempts. Proceeding with available logs.")
            collected_logs = logs

        # Modified to focus ONLY on user messages, filtering out trigger phrases
        conversation_str = "IMPORTANT: This analysis is based ONLY on the following USER MESSAGES:\n\n"
        for log_item in collected_logs:
            user_text = log_item["request"]["input"].get("text", "").strip()
            
            # Skip empty messages
            if not user_text:
                continue
                
            # Check if message contains any filter phrases and remove them
            clean_text = user_text
            for phrase in filter_phrases:
                if phrase.lower() in clean_text.lower():
                    # Remove the phrase and clean up extra spaces
                    clean_text = clean_text.replace(phrase, "").strip()
                    clean_text = re.sub(r'\s+', ' ', clean_text)
            
            # Skip if message is now empty after filtering or contains only punctuation
            if not clean_text or clean_text.strip('.,!? ') == '':
                continue
                
            # Add only the user message to the transcript
            conversation_str += f"USER: {clean_text}\n\n"
            
        print("Conversation transcript constructed with focus on user messages only.")
        return conversation_str

    except Exception as e:
        print(f"Error retrieving conversation: {e}")
        return "Error retrieving conversation."

def call_local_llm(prompt, model=LLM_MODEL, max_tokens=3000, interaction_type="general"):
    """Call local LLM via HTTP API with comprehensive logging"""
    print(f"Calling local LLM for {interaction_type}...")
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        response_content = result.get("message", {}).get("content", "").strip()
        
        # Log this interaction
        log_llm_interaction(prompt, response_content, interaction_type)
        
        return response_content
    except requests.exceptions.RequestException as e:
        print(f"LLM API error: {e}")
        # Log the error as well
        log_llm_interaction(prompt, f"Error: {str(e)}", f"error_{interaction_type}")
        return f"Error calling local LLM: {e}"

def combine_docs(docs):
    """Combine document chunks into single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question, retriever_or_text, interaction_type="rag_query"):
    """RAG pipeline for question answering with enhanced logging"""
    if retriever_or_text is None:
        return "Error: Document retriever not available."
    
    if isinstance(retriever_or_text, str):
        # Change the order: Context (text) comes first, then Question (prompt words)
        return call_local_llm(
            f"Context: {retriever_or_text}\n\nQuestion: {question}", 
            interaction_type=interaction_type
        )
    else:
        # Log retrieval method being used
        if USE_MMR:
            print(f"Using MMR retrieval for {interaction_type}")
        else:
            print(f"Using standard retrieval for {interaction_type}")
        
        # Get documents
        retrieved_docs = retriever_or_text.invoke(question)
        print(f"Retrieved {len(retrieved_docs)} documents for {interaction_type}")
        
        context = combine_docs(retrieved_docs)
        # Now, context comes first and question afterwards
        return call_local_llm(
            f"Context: {context}\n\nQuestion: {question}",
            interaction_type=interaction_type
        )

# Load prompt definitions from the prompts.json file
try:
    with open("prompts.json", "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
    print("Prompts loaded successfully.")
except Exception as e:
    print(f"Error loading prompts: {e}")
    PROMPTS = {}

def run_most_analysis(retriever_or_text, conversation_text):
    most_prompt = PROMPTS.get("most")
    if not most_prompt:
        return "Error: MOST prompt not found."
    
    # Get the raw text analysis
    result = rag_chain(most_prompt, retriever_or_text, interaction_type="most_analysis")
    
    # Try to extract MOST data and generate a chart
    try:
        # Check if the result contains MOST_ANALYSIS tags
        if "<MOST_ANALYSIS>" in result and "</MOST_ANALYSIS>" in result:
            # Parse the MOST analysis data
            most_data = parse_most_analysis(result)
            
            if most_data and len(most_data) > 0:
                # Generate and save the chart
                chart_path = save_most_chart(most_data)
                
                # If successful, add the chart path to the result
                if chart_path:
                    result += f"\n\n<MOST_CHART>{chart_path}</MOST_CHART>"
                    print(f"Generated MOST chart: {chart_path}")
    except Exception as e:
        print(f"Error during MOST chart generation: {e}")
    
    return result

def parse_most_analysis(input_text):
    # Remove the MOST_ANALYSIS tags from the input.
    cleaned_text = re.sub(r"</?MOST_ANALYSIS>", "", input_text, flags=re.IGNORECASE).strip()
    lines = cleaned_text.splitlines()
    
    parsed_data = {}
    current_category = None
    category_lines = []
    
    for line in lines:
        # Remove trailing whitespace while preserving blank lines.
        line = line.rstrip()
        # Detect category headers (e.g., "- Mission:").
        header_match = re.match(r"-\s*(\w+):", line)
        if header_match:
            if current_category is not None:
                parsed_data[current_category] = "\n".join(category_lines)
                category_lines = []
            current_category = header_match.group(1)
        else:
            category_lines.append(line)
    
    if current_category is not None:
        parsed_data[current_category] = "\n".join(category_lines)
    
    return parsed_data

def create_most_figure(most_data):
    """
    Create a composite figure illustrating the MOST analysis categories as horizontal slices.
    
    Each slice (Tactics, Strategy, Objectives, Mission) is represented by:
      - A left-hand polygonal background.
      - A right-hand area where a fixed-width color block appears on the left,
        with the descriptive text to its right.
    
    Horizontal separator lines are added between text sections.
    """
    # Local helper functions to define the boundaries of the left and right edges.
    def boundary_left(y):
        return (0.5 / 4.0) * y

    def boundary_right(y):
        return 1.0 - (y / 8.0)
    
    # Predefine colors and category order.
    slice_colors = ["#7F8C8D", "#3498DB", "#E67E22", "#E74C3C"]
    category_order = ["Tactics", "Strategy", "Objectives", "Mission"]
    
    # Initialize the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    
    # Draw polygonal slices.
    for i in range(4):
        y_bottom, y_top = i, i + 1
        polygon_coords = [
            (boundary_left(y_bottom), y_bottom / 4.0),
            (boundary_left(y_top),    y_top / 4.0),
            (boundary_right(y_top),   y_top / 4.0),
            (boundary_right(y_bottom), y_bottom / 4.0)
        ]
        slice_polygon = Polygon(polygon_coords, closed=True, facecolor=slice_colors[i],
                                  edgecolor='white', linewidth=2)
        ax.add_patch(slice_polygon)
        
        label_y = (y_bottom / 4.0) + 0.05
        x_mid = (boundary_left(y_bottom) + boundary_right(y_bottom)) / 2.0
        ax.text(x_mid, label_y, category_order[i],
                ha='center', va='top', color='white', fontsize=18, fontweight='bold')
    
    # Parameters for wrapping and scaling.
    WRAP_WIDTH = 100
    BASE_FONT_SIZE = 12
    BASE_LINE_HEIGHT = 0.08
    AVAILABLE_HEIGHT = 0.8
    
    def wrap_text_preserving_format(text, width):
        wrapped_lines = []
        for line in text.splitlines():
            if line.strip() == "":
                wrapped_lines.append("")
                continue
            stripped_line = line.lstrip()
            if stripped_line.startswith("·"):
                content = stripped_line[1:].strip()
                wrapped = textwrap.fill(content, width=width,
                                        initial_indent="· ", subsequent_indent="  ")
            elif stripped_line.startswith("-"):
                content = stripped_line[1:].strip()
                wrapped = textwrap.fill(content, width=width,
                                        initial_indent="- ", subsequent_indent="  ")
            else:
                wrapped = textwrap.fill(line, width=width)
            wrapped_lines.extend(wrapped.splitlines())
        return wrapped_lines

    # Wrap text and compute scaling.
    global_scale = 1.0
    wrapped_texts = {}
    for category in category_order:
        description = most_data.get(category, "")
        wrapped_lines = wrap_text_preserving_format(description, WRAP_WIDTH)
        wrapped_texts[category] = wrapped_lines
        num_lines = len(wrapped_lines)
        text_tile_height = num_lines * BASE_LINE_HEIGHT
        tile_scale = 1.0 if text_tile_height <= AVAILABLE_HEIGHT else AVAILABLE_HEIGHT / text_tile_height
        global_scale = min(global_scale, tile_scale)
    
    global_font_size = round(BASE_FONT_SIZE * global_scale)
    
    # Compute a consistent text box width.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_text_width = 0
    for category in category_order:
        text_content = "\n".join(wrapped_texts.get(category, ""))
        temp_text = ax.text(0, 0, text_content, fontsize=global_font_size, ha='left', va='center')
        bbox = temp_text.get_window_extent(renderer=renderer)
        inv_transform = ax.transData.inverted()
        bbox_data = inv_transform.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        width_data = bbox_data[1][0] - bbox_data[0][0]
        max_text_width = max(max_text_width, width_data)
        temp_text.remove()
    
    PAD_X = 0.1
    COLOR_BLOCK_WIDTH = 0.05  # Fixed width for the color block.
    total_text_width = COLOR_BLOCK_WIDTH + PAD_X + max_text_width
    
    # Render the color blocks and text.
    text_rectangles = []
    for i, category in enumerate(category_order):
        tile_y_bottom = i / 4.0
        tile_height = 1 / 4.0
        text_content = "\n".join(wrapped_texts.get(category, ""))
        
        color_block = Rectangle((1.05, tile_y_bottom), COLOR_BLOCK_WIDTH, tile_height,
                                  facecolor=slice_colors[i], alpha=0.2, edgecolor='none')
        ax.add_patch(color_block)
        text_rectangles.append((1.05, tile_y_bottom, total_text_width, tile_height))
        
        ax.text(1.05 + COLOR_BLOCK_WIDTH + PAD_X, tile_y_bottom + tile_height / 2.0, text_content,
                ha='left', va='center', color='black', fontsize=global_font_size, zorder=3)
    
    # Draw separator lines.
    for (x, y, width, height) in text_rectangles[:-1]:
        ax.plot([x, x + width], [y + height, y + height], color='white', linewidth=2)
    
    # Instead of calling plt.show(), return the figure object.
    return fig

def save_most_chart(most_dict):
    try:
        # Create a directory for analysis images if it doesn't exist
        images_dir = os.path.join('static', 'analysis_images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"most_analysis_{timestamp}.png"
        file_path = os.path.join(images_dir, filename)
        
        # Generate the figure
        fig = create_most_figure(most_dict)
        
        # Save the figure to a file
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
        
        # Return the path relative to static folder
        return os.path.join('analysis_images', filename)
    except Exception as e:
        print(f"Error generating MOST chart: {e}")
        return None

def run_swot_analysis(retriever_or_text, conversation_text):
    swot_prompt = PROMPTS.get("swot")
    if not swot_prompt:
        return "Error: SWOT prompt not found."
    
    # Get the raw text analysis
    result = rag_chain(swot_prompt, retriever_or_text, interaction_type="swot_analysis")
    
    # Try to extract SWOT data and generate a chart
    try:
        # Check if the result contains SWOT_ANALYSIS tags
        if "<SWOT_ANALYSIS>" in result and "</SWOT_ANALYSIS>" in result:
            # Parse the SWOT analysis data
            swot_data = parse_swot_analysis(result)
            
            if swot_data and all(len(swot_data[cat]) > 0 for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]):
                # Generate and save the chart
                chart_path = save_swot_chart(swot_data)
                
                # If successful, add the chart path to the result
                if chart_path:
                    result += f"\n\n<SWOT_CHART>{chart_path}</SWOT_CHART>"
                    print(f"Generated SWOT chart: {chart_path}")
    except Exception as e:
        print(f"Error during SWOT chart generation: {e}")
    
    return result

def parse_swot_analysis(swot_text: str):
    # Remove the tags if present
    swot_text = re.sub(r"</?SWOT_ANALYSIS>", "", swot_text).strip()
    
    categories = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
    swot_data = {cat: [] for cat in categories}
    
    current_cat = None
    current_bullet = None
    
    for line in swot_text.splitlines():
        line = line.strip()
        if not line:
            continue
        header_match = re.match(r"^- (\w+):", line)
        if header_match:
            cat = header_match.group(1)
            if cat in categories:
                if current_bullet:
                    swot_data[current_cat].append(current_bullet.strip())
                    current_bullet = None
                current_cat = cat
            continue
        
        if line.startswith("·") or line.startswith("•") or line.startswith("-"):
            if current_bullet:
                swot_data[current_cat].append(current_bullet.strip())
            current_bullet = re.sub(r"^[·•\-]\s*", "", line)
        else:
            if current_bullet is not None:
                current_bullet += " " + line
    if current_bullet:
        swot_data[current_cat].append(current_bullet.strip())
    
    return swot_data

def create_swot_figure(swot_data):
    # Create a square figure with high resolution and a constrained layout
    fig = plt.figure(figsize=(10, 10), dpi=300, constrained_layout=True)
    
    # --- Left Margin ---
    ax_left_top = fig.add_axes([0.05, 0.375, 0.1, 0.375])
    ax_left_top.set_facecolor('#FFEB3B')  # yellow
    ax_left_top.set_xticks([]); ax_left_top.set_yticks([])
    for spine in ax_left_top.spines.values():
        spine.set_visible(False)
    ax_left_top.text(
        0.5, 0.5,
        "Internal origin\n(attributes of the organization)",
        ha='center', va='center', rotation=90, color='black', fontsize=10,
        transform=ax_left_top.transAxes
    )
    
    ax_left_bottom = fig.add_axes([0.05, 0.0, 0.1, 0.375])
    ax_left_bottom.set_facecolor('#64B5F6')  # blue
    ax_left_bottom.set_xticks([]); ax_left_bottom.set_yticks([])
    for spine in ax_left_bottom.spines.values():
        spine.set_visible(False)
    ax_left_bottom.text(
        0.5, 0.5,
        "External origin\n(attributes of the environment)",
        ha='center', va='center', rotation=90, color='black', fontsize=10,
        transform=ax_left_bottom.transAxes
    )
    
    # --- Top Margin ---
    ax_top_green = fig.add_axes([0.15, 0.75, 0.425, 0.1])
    ax_top_green.set_facecolor('#8BC34A')  # green
    ax_top_green.set_xticks([]); ax_top_green.set_yticks([])
    for spine in ax_top_green.spines.values():
        spine.set_visible(False)
    ax_top_green.text(
        0.5, 0.5,
        "Helpful\nto achieving the objective",
        ha='center', va='center', color='black', fontsize=12,
        transform=ax_top_green.transAxes
    )
    
    ax_top_red = fig.add_axes([0.575, 0.75, 0.425, 0.1])
    ax_top_red.set_facecolor('#F44336')  # red
    ax_top_red.set_xticks([]); ax_top_red.set_yticks([])
    for spine in ax_top_red.spines.values():
        spine.set_visible(False)
    ax_top_red.text(
        0.5, 0.5,
        "Harmful\nto achieving the objective",
        ha='center', va='center', color='black', fontsize=12,
        transform=ax_top_red.transAxes
    )
    
    # --- Center 2×2 Grid ---
    ax_s = fig.add_axes([0.15, 0.375, 0.425, 0.375])
    ax_s.set_facecolor('#DCE775')  # light lime green
    ax_s.set_xticks([]); ax_s.set_yticks([])
    for spine in ax_s.spines.values():
        spine.set_visible(False)
    
    ax_w = fig.add_axes([0.575, 0.375, 0.425, 0.375])
    ax_w.set_facecolor('#FFCDD2')  # light red/pink
    ax_w.set_xticks([]); ax_w.set_yticks([])
    for spine in ax_w.spines.values():
        spine.set_visible(False)
    
    ax_o = fig.add_axes([0.15, 0.0, 0.425, 0.375])
    ax_o.set_facecolor('#B2DFDB')  # teal-ish
    ax_o.set_xticks([]); ax_o.set_yticks([])
    for spine in ax_o.spines.values():
        spine.set_visible(False)
    
    ax_t = fig.add_axes([0.575, 0.0, 0.425, 0.375])
    ax_t.set_facecolor('#E1BEE7')  # light purple
    ax_t.set_xticks([]); ax_t.set_yticks([])
    for spine in ax_t.spines.values():
        spine.set_visible(False)
    
    def plot_bullets(ax, bullet_list, wrap_width=50,
                     base_font_size=10, base_line_height=0.05, available_space=0.8):
        # Pre-wrap all bullet text lines
        wrapped_bullets = []
        total_lines = 0
        for bullet in bullet_list:
            wrapped = textwrap.wrap(bullet, width=wrap_width)
            wrapped_bullets.append(wrapped)
            # Each bullet contributes its wrapped lines plus one extra line for spacing.
            total_lines += len(wrapped) + 1

        # Determine if scaling is needed
        scale = 1.0
        if total_lines * base_line_height > available_space:
            scale = available_space / (total_lines * base_line_height)
        font_size = base_font_size * scale
        line_height = base_line_height * scale

        y = 0.8  # Starting y-coordinate (normalized)
        for wrapped in wrapped_bullets:
            if not wrapped:
                continue
            # Draw first line with bullet marker
            ax.text(0.05, y, f"• {wrapped[0]}",
                    fontsize=font_size, ha='left', va='top', transform=ax.transAxes)
            y -= line_height
            # Draw any additional lines with indent
            for line in wrapped[1:]:
                ax.text(0.1, y, line,
                        fontsize=font_size, ha='left', va='top', transform=ax.transAxes)
                y -= line_height
            # Extra spacing between bullet points
            y -= line_height
            if y < 0.05:
                break  # Stop if vertical space is nearly exhausted

    # --- Quadrant: Strengths ---
    ax_s.text(0.5, 0.5, "S", fontsize=80, color='white', alpha=0.3,
              ha='center', va='center', transform=ax_s.transAxes)
    ax_s.text(0.5, 0.9, "Strengths", fontsize=14, weight='bold',
              ha='center', va='top', transform=ax_s.transAxes)
    plot_bullets(ax_s, swot_data["Strengths"])
    
    # --- Quadrant: Weaknesses ---
    ax_w.text(0.5, 0.5, "W", fontsize=80, color='white', alpha=0.3,
              ha='center', va='center', transform=ax_w.transAxes)
    ax_w.text(0.5, 0.9, "Weaknesses", fontsize=14, weight='bold',
              ha='center', va='top', transform=ax_w.transAxes)
    plot_bullets(ax_w, swot_data["Weaknesses"])
    
    # --- Quadrant: Opportunities ---
    ax_o.text(0.5, 0.5, "O", fontsize=80, color='white', alpha=0.3,
              ha='center', va='center', transform=ax_o.transAxes)
    ax_o.text(0.5, 0.9, "Opportunities", fontsize=14, weight='bold',
              ha='center', va='top', transform=ax_o.transAxes)
    plot_bullets(ax_o, swot_data["Opportunities"])
    
    # --- Quadrant: Threats ---
    ax_t.text(0.5, 0.5, "T", fontsize=80, color='white', alpha=0.3,
              ha='center', va='center', transform=ax_t.transAxes)
    ax_t.text(0.5, 0.9, "Threats", fontsize=14, weight='bold',
              ha='center', va='top', transform=ax_t.transAxes)
    plot_bullets(ax_t, swot_data["Threats"])
    
    return fig

def save_swot_chart(swot_dict):
    try:
        # Create a directory for analysis images if it doesn't exist
        images_dir = os.path.join('static', 'analysis_images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"swot_analysis_{timestamp}.png"
        file_path = os.path.join(images_dir, filename)
        
        # Generate the figure
        fig = create_swot_figure(swot_dict)
        
        # Save the figure to a file
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
        
        # Return the path relative to static folder
        return os.path.join('analysis_images', filename)
    except Exception as e:
        print(f"Error generating SWOT chart: {e}")
        return None

def run_pestle_analysis(retriever_or_text, conversation_text):
    pestle_prompt = PROMPTS.get("pestle")
    if not pestle_prompt:
        return "Error: PESTLE prompt not found."
    
    # Get the raw text analysis
    result = rag_chain(pestle_prompt, retriever_or_text, interaction_type="pestle_analysis")
    
    # Try to extract PESTLE data and generate a chart
    try:
        # Check if the result contains PESTLE_ANALYSIS tags
        if "<PESTLE_ANALYSIS>" in result and "</PESTLE_ANALYSIS>" in result:
            # Parse the PESTLE analysis data
            pestle_data = parse_pestle_analysis(result)
            
            if pestle_data and len(pestle_data) > 0:
                # Generate and save the chart
                chart_path = save_pestle_chart(pestle_data)
                
                # If successful, add the chart path to the result
                if chart_path:
                    result += f"\n\n<PESTLE_CHART>{chart_path}</PESTLE_CHART>"
                    print(f"Generated PESTLE chart: {chart_path}")
    except Exception as e:
        print(f"Error during PESTLE chart generation: {e}")
    
    return result

def parse_pestle_analysis(text):
    # Remove the <PESTLE_ANALYSIS> tags
    text = re.sub(r"</?PESTLE_ANALYSIS>", "", text, flags=re.IGNORECASE).strip()
    lines = text.splitlines()
    
    categories = {}
    current_cat = None
    bullets = []
    current_bullet = None

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        # Check for a new category indicator (e.g., "- Political:")
        cat_match = re.match(r"-\s+(\w+):", line)
        if cat_match:
            # Finalize the previous category if one exists
            if current_cat is not None:
                if current_bullet is not None:
                    bullets.append(current_bullet.strip())
                    current_bullet = None
                categories[current_cat] = bullets
            # Start a new category
            current_cat = cat_match.group(1)
            bullets = []
            current_bullet = None
        # Check for a bullet marker at the beginning of the line (using common markers)
        elif line.startswith("·") or line.startswith("•") or line.startswith("-"):
            # If there's an ongoing bullet, save it first.
            if current_bullet is not None:
                bullets.append(current_bullet.strip())
            # Start a new bullet point, removing the bullet marker and extra whitespace.
            current_bullet = re.sub(r"^[·•\-]\s*", "", line)
        else:
            # If not a bullet marker, assume it's a continuation of the current bullet.
            if current_bullet is not None:
                current_bullet += " " + line
            else:
                current_bullet = line

    # Finalize the last bullet and category, if any.
    if current_bullet is not None:
        bullets.append(current_bullet.strip())
    if current_cat is not None:
        categories[current_cat] = bullets
    
    return categories

def draw_pestle_figure(pestle_dict):
    # Academic style fonts
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titleweight': 'bold'
    })
    
    # Predefined style for each PESTLE category
    category_styles = {
        "Political": {"letter": "P", "color": "#8C8C8C"},      # Gray
        "Economic": {"letter": "E", "color": "#DD8452"},       # Orange
        "Social": {"letter": "S", "color": "#4C72B0"},         # Blue
        "Technological": {"letter": "T", "color": "#8172B3"},  # Muted Purple
        "Legal": {"letter": "L", "color": "#C44E52"},          # Red
        "Environmental": {"letter": "E", "color": "#55A868"}   # Green
    }   
    
    # Order categories: first the typical PESTLE order, then any extras.
    pestle_order = ["Political", "Economic", "Social", "Technological", "Legal", "Environmental"]
    cat_in_order = [cat for cat in pestle_order if cat in pestle_dict]
    other_cats = [cat for cat in pestle_dict if cat not in pestle_order]
    final_cats = cat_in_order + other_cats
    
    # Pre-calculate the required block height for each category.
    base_line_height = 0.05  
    min_block_height = 0.2     
    extra_padding = 0.1        
    wrap_width = 100           
    
    block_heights = {}
    for cat in final_cats:
        total_lines = 0
        for bullet in pestle_dict[cat]:
            wrapped = textwrap.wrap(bullet, width=wrap_width)
            if wrapped:
                total_lines += len(wrapped) + 1  # +1 for a blank line between bullets
        required_height = max(min_block_height, total_lines * base_line_height + extra_padding)
        block_heights[cat] = required_height

    total_height_units = sum(block_heights[cat] for cat in final_cats)
    
    # Create a figure. We'll use a fixed width (e.g., 10 inches) and a height scaled by total_height_units.
    fig = plt.figure(figsize=(10, 10), dpi=300)
    
    # Instead of starting from the bottom, we start from the top (y=1) and work down.
    current_top = 1  
    for cat in final_cats:
        rel_height = block_heights[cat] / total_height_units  
        bottom = current_top - rel_height
        ax = fig.add_axes([0, bottom, 1, rel_height])
        ax.axis('off')
        
        left_bar_width = 0.18  
        style = category_styles.get(cat, {"letter": cat[0], "color": "#666666"})
        letter = style["letter"]
        color = style["color"]
        
        ax.add_patch(
            Rectangle(
                (0, 0),         
                left_bar_width, 
                1,              
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                transform=ax.transAxes,
                zorder=1
            )
        )
        
        ax.text(
            0.03, 0.55,
            letter,
            color="white",
            fontsize=28,
            fontweight="bold",
            ha="left",
            va="center",
            transform=ax.transAxes,
            zorder=2
        )
        ax.text(
            0.03, 0.35,
            cat,
            color="white",
            fontsize=12,
            fontweight="normal",
            ha="left",
            va="center",
            transform=ax.transAxes,
            zorder=2
        )
        
        ax.add_patch(
            Rectangle(
                (left_bar_width, 0),
                1 - left_bar_width,
                1,
                facecolor=color,
                alpha=0.2,
                transform=ax.transAxes,
                zorder=1
            )
        )
        
        combined_text = ""
        for bullet in pestle_dict[cat]:
            wrapped_lines = textwrap.wrap(bullet, width=wrap_width)
            if wrapped_lines:
                combined_text += f"• {wrapped_lines[0]}\n"
                for wl in wrapped_lines[1:]:
                    combined_text += f"  {wl}\n"
                combined_text += "\n"
        
        ax.text(
            left_bar_width + 0.02, 0.5,
            combined_text.strip(),
            color="black",
            fontsize=10,
            ha="left",
            va="center",
            transform=ax.transAxes,
            zorder=2
        )
        
        # Update current_top for the next block.
        current_top = bottom
    
    return fig

def save_pestle_chart(pestle_dict):
    try:
        # Create a directory for analysis images if it doesn't exist
        images_dir = os.path.join('static', 'analysis_images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pestle_analysis_{timestamp}.png"
        file_path = os.path.join(images_dir, filename)
        
        # Generate the figure
        fig = draw_pestle_figure(pestle_dict)
        
        # Save the figure to a file
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
        
        # Return the path relative to static folder
        return os.path.join('analysis_images', filename)
    except Exception as e:
        print(f"Error generating PESTLE chart: {e}")
        return None

def run_sentiment_analysis(retriever_or_text, conversation_text):
    sentiment_prompt = PROMPTS.get("sentiment")
    if not sentiment_prompt:
        return "Error: Sentiment prompt not found."
    
    # Get the raw text analysis
    result = rag_chain(sentiment_prompt, retriever_or_text, interaction_type="sentiment_analysis")
    
    try:
        # Get raw text for word cloud and sentiment analysis
        file_path = None
        if isinstance(retriever_or_text, str):
            raw_text = retriever_or_text
        else:
            # Try to get the file path from the process_pdf function context
            if hasattr(retriever_or_text, '_file_path'):
                file_path = retriever_or_text._file_path
            
            # If we have a file path, extract text directly
            if file_path and os.path.exists(file_path):
                raw_text = extract_text_from_pdf(file_path)
            else:
                # Fallback: use context from retriever
                try:
                    docs = retriever_or_text.invoke("Give me the full document content")
                    raw_text = combine_docs(docs)
                except:
                    # Last resort: just use the analysis result
                    raw_text = result
        
        # Create directory for analysis images if it doesn't exist
        images_dir = os.path.join('static', 'analysis_images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate and save word cloud
        wordcloud_filename = f"wordcloud_{timestamp}.png"
        wordcloud_path = os.path.join(images_dir, wordcloud_filename)
        
        saved_wordcloud_path = save_wordcloud(raw_text, wordcloud_path)
        if saved_wordcloud_path:
            wordcloud_relative_path = os.path.join('analysis_images', wordcloud_filename)
            print(f"Generated word cloud: {wordcloud_relative_path}")
        else:
            wordcloud_relative_path = None
        
        # Clean and chunk text for sentiment analysis
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text, chunk_size=512, overlap=50)
        
        # Create sentiment pipeline and analyze
        sentiment_pipe = create_sentiment_pipeline()
        sentiment_score = analyze_chunks_with_pipeline(chunks, sentiment_pipe)
        print(f"Calculated sentiment score: {sentiment_score:.2f}")
        
        # Generate and save sentiment visualization
        sentiment_viz_filename = f"sentiment_{timestamp}.png"
        sentiment_viz_path = os.path.join(images_dir, sentiment_viz_filename)
        
        saved_sentiment_path = save_sentiment_visualization(sentiment_score, sentiment_viz_path)
        if saved_sentiment_path:
            sentiment_viz_relative_path = os.path.join('analysis_images', sentiment_viz_filename)
            print(f"Generated sentiment visualization: {sentiment_viz_relative_path}")
        else:
            sentiment_viz_relative_path = None
        
        # Add visualization paths to result
        if wordcloud_relative_path:
            result += f"\n\n<SENTIMENT_WORDCLOUD>{wordcloud_relative_path}</SENTIMENT_WORDCLOUD>"
        if sentiment_viz_relative_path:
            result += f"\n\n<SENTIMENT_VISUALIZATION>{sentiment_viz_relative_path}</SENTIMENT_VISUALIZATION>"
            
    except Exception as e:
        print(f"Error during sentiment visualization generation: {e}")
    
    return result

# -------------------------------------------
# Text Cleaning and Processing
# -------------------------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    # Remove punctuation using translate
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_val = " ".join(chunk_words)
        chunks.append(chunk_text_val)
        start += (chunk_size - overlap)
    return chunks

# -------------------------------------------
# Word Cloud Generation
# -------------------------------------------
def get_optimized_stopwords():
    stopwords = set(STOPWORDS)
    # Additional stopwords common in annual reports
    additional_stopwords = {
        "annual", "report", "company", "inc", "corp", "corporation", "limited", "llc", "group",
        "shareholder", "shareholders", "board", "management", "fiscal", "year", "financial",
        "statement", "statements", "revenue", "profit", "loss", "assets", "liabilities", "balance",
        "sheet", "audit", "million", "billion", "usd", "ebitda", "operating", "cash", "flow",
        "chairman", "ceo", "director", "chief", "executive", "officer", "investment", "risk",
        "stock", "dividend", "growth", "business", "industry", "market", "strategy", "performance",
        "activities", "products", "services", "development", "reporting", "overview", "results",
        "improvement", "increase", "decrease", "subsidiary", "subsidiaries", "quarter", "review",
        "governance", "compliance", "regulatory", "notice", "interim", "discussion", "analysis",
        "consolidated", "unit", "units", "directors", "executives", "employees", "employee",
        "workforce", "infrastructure", "cost", "costs", "factors", "success", "achieve",
        "achieved", "achieving", "goals", "objective", "objectives", "plans", "initiatives",
        "committee", "committees", "agenda", "financials", "conclusion", "notes", "disclaimer",
        "non", "gaap", "forward", "looking", "information", "advisory", "proposal", "proxy",
        "meeting", "accounting", "material", "component", "segment", "segments", "operation",
        "operations", "operational"
    }
    stopwords.update(additional_stopwords)
    return stopwords

def create_wordcloud(text):
    stopwords = get_optimized_stopwords()
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        collocations=False,
        max_words=200,
        max_font_size=100
    ).generate(text)
    
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    return fig

def save_wordcloud(text, file_path):
    try:
        fig = create_wordcloud(text)
        fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return file_path
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

# -------------------------------------------
# Sentiment Analysis Visualization
# -------------------------------------------
def create_sentiment_pipeline():
    """
    Create and return an instance of NLTK's SentimentIntensityAnalyzer.
    """
    return SentimentIntensityAnalyzer()

def analyze_chunks_with_pipeline(chunks, sentiment_analyzer):
    """
    Analyze each text chunk using NLTK's VADER analyzer.
    Returns the average compound sentiment score from all chunks.
    """
    scores = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        # Get sentiment scores for the chunk
        sentiment_result = sentiment_analyzer.polarity_scores(chunk)
        compound = sentiment_result['compound']  # Compound score in range [-1, 1]
        scores.append(compound)
    return sum(scores) / len(scores) if scores else 0.0

def create_sentiment_figure(sentiment_score):
    # Ensure score is in range [-1, 1]
    score = max(min(sentiment_score, 1.0), -1.0)
    
    # Set up directories and paths
    images_dir = os.path.join('static', 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    base_image_path = os.path.join(images_dir, 'emoji_scale.jpg')
    
    # If the emoji scale image doesn't exist, create a fallback
    if not os.path.exists(base_image_path):
        print(f"Warning: Emoji scale image not found at {base_image_path}")
        # Create a fallback gradient image
        fig, ax = plt.subplots(figsize=(10, 3))
        gradient = np.linspace(-1, 1, 100)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap=plt.cm.coolwarm, extent=[-1, 1, 0, 1])
        ax.text(-0.95, 0.5, "Negative", fontsize=14, va='center', ha='left')
        ax.text(0.95, 0.5, "Positive", fontsize=14, va='center', ha='right')
        ax.text(0, 0.5, "Neutral", fontsize=14, va='center', ha='center')
        ax.set_yticks([])
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        
        # Add arrow marker at score position
        ax.plot([score], [0.5], 'v', color='red', markersize=12)
        ax.text(score, 0.7, f"{score:.2f}", ha='center', fontsize=12, color='black')
        
        return fig
    
    try:
        # Open the emoji scale image
        print(f"Using emoji scale image from {base_image_path}")
        img = Image.open(base_image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        
        # Calculate arrow position based on sentiment score
        width, height = img.size
        # MODIFIED: Use middle 90% of width instead of full width
        leftmost_pos = int(width * 0.05)  # 5% from left
        rightmost_pos = int(width * 0.95)  # 95% from left (5% from right)
        usable_width = rightmost_pos - leftmost_pos
        normalized_score = (score + 1) / 2.0  # Convert from [-1,1] to [0,1]
        x_pos = leftmost_pos + int(normalized_score * usable_width)
        
        # Draw the arrow
        arrow_tip_y = int(height * 0.64)
        arrow_base_y = int(height * 0.68)
        arrow_width = 6
        arrow_polygon = [
            (x_pos, arrow_tip_y),
            (x_pos - arrow_width, arrow_base_y),
            (x_pos + arrow_width, arrow_base_y),
        ]
        
        draw.polygon(arrow_polygon, fill="red", outline="black")
        
        # Try to create a font - if system font not available, use default
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            try:
                # Try a more universal font that might be available on Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except IOError:
                # Final fallback to default
                font = ImageFont.load_default()
        
        # Add score text
        text_position = (x_pos - 15, arrow_base_y + 10)
        draw.text(text_position, f"{score:.2f}", fill="black", font=font)
        
        # Convert to numpy array for matplotlib
        img_array = np.array(img)
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img_array)
        ax.axis('off')
        
        return fig
        
    except Exception as e:
        print(f"Error creating sentiment figure with local image: {e}")
        # Fall back to a simple matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 3))
        gradient = np.linspace(-1, 1, 100)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap=plt.cm.RdYlGn, extent=[-1, 1, 0, 1])
        ax.text(-0.95, 0.5, "Negative", fontsize=14, va='center', ha='left')
        ax.text(0.95, 0.5, "Positive", fontsize=14, va='center', ha='right')
        ax.text(0, 0.5, "Neutral", fontsize=14, va='center', ha='center')
        
        # Add arrow marker at score position
        ax.plot([score], [0.5], 'v', color='red', markersize=12)
        ax.text(score, 0.7, f"{score:.2f}", ha='center', fontsize=12, color='black')
        
        ax.set_yticks([])
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        
        return fig

def save_sentiment_visualization(sentiment_score, file_path):
    try:
        fig = create_sentiment_figure(sentiment_score)
        fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return file_path
    except Exception as e:
        print(f"Error generating sentiment visualization: {e}")
        return None

# -------------------------------------------
# Text Extraction Utility
# -------------------------------------------
def extract_text_from_pdf(pdf_path):
    text_content = []
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text_content.append(page.extract_text())
    return " ".join(text_content)

def run_additional_analysis(retriever_or_text, conversation_text, analysis_type):
    additional_prompt_template = PROMPTS.get("additional")
    if not additional_prompt_template:
        return "Error: Additional analysis prompt not found."
    additional_prompt = additional_prompt_template.format(analysis_type=analysis_type)
    return rag_chain(additional_prompt, retriever_or_text, interaction_type="additional_analysis")

# Flask routes
@app.route('/health')
def health_check():
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        ollama_status = "running" if response.status_code == 200 else "error"
    except:
        ollama_status = "not running"

    return jsonify({
        "status": "healthy",
        "api": "running",
        "ollama": ollama_status,
        "mmr_enabled": USE_MMR
    })

# New endpoint to toggle MMR retrieval
@app.route('/toggle-mmr', methods=["POST"])
def toggle_mmr():
    global USE_MMR
    data = request.json
    if 'enabled' in data:
        USE_MMR = bool(data['enabled'])
        return jsonify({
            "success": True,
            "mmr_enabled": USE_MMR,
            "msg": f"MMR retrieval {'enabled' if USE_MMR else 'disabled'}"
        })
    return jsonify({"success": False, "msg": "Missing 'enabled' parameter"})

# New endpoints for sequential analysis with MMR
@app.route('/analyze/most', methods=["POST"])
def analyze_most():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"msg": "Invalid file path", "success": False})

    _, _, retriever_or_text = process_pdf(file_path)
    conversation_text = get_conversation_text()
    
    result = run_most_analysis(retriever_or_text, conversation_text)
    
    # Check if we have a chart path in the result
    chart_path = None
    chart_match = re.search(r"<MOST_CHART>(.*?)</MOST_CHART>", result)
    if chart_match:
        chart_path = chart_match.group(1)
        # Remove the chart path tags from the text result
        result = re.sub(r"<MOST_CHART>.*?</MOST_CHART>", "", result)
    
    return jsonify({
        "msg": result, 
        "analysis_type": "most", 
        "success": True, 
        "mmr_used": USE_MMR,
        "has_visualization": chart_path is not None,
        "visualization_path": chart_path
    })

@app.route('/analyze/swot', methods=["POST"])
def analyze_swot():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"msg": "Invalid file path", "success": False})

    _, _, retriever_or_text = process_pdf(file_path)
    conversation_text = get_conversation_text()
    
    result = run_swot_analysis(retriever_or_text, conversation_text)
    
    # Check if we have a chart path in the result
    chart_path = None
    chart_match = re.search(r"<SWOT_CHART>(.*?)</SWOT_CHART>", result)
    if chart_match:
        chart_path = chart_match.group(1)
        # Remove the chart path tags from the text result
        result = re.sub(r"<SWOT_CHART>.*?</SWOT_CHART>", "", result)
    
    return jsonify({
        "msg": result, 
        "analysis_type": "swot", 
        "success": True, 
        "mmr_used": USE_MMR,
        "has_visualization": chart_path is not None,
        "visualization_path": chart_path
    })

@app.route('/analyze/pestle', methods=["POST"])
def analyze_pestle():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"msg": "Invalid file path", "success": False})

    _, _, retriever_or_text = process_pdf(file_path)
    conversation_text = get_conversation_text()
    
    result = run_pestle_analysis(retriever_or_text, conversation_text)
    
    # Check if we have a chart path in the result
    chart_path = None
    chart_match = re.search(r"<PESTLE_CHART>(.*?)</PESTLE_CHART>", result)
    if chart_match:
        chart_path = chart_match.group(1)
        # Remove the chart path tags from the text result
        result = re.sub(r"<PESTLE_CHART>.*?</PESTLE_CHART>", "", result)
    
    return jsonify({
        "msg": result, 
        "analysis_type": "pestle", 
        "success": True, 
        "mmr_used": USE_MMR,
        "has_visualization": chart_path is not None,
        "visualization_path": chart_path
    })

@app.route('/analyze/sentiment', methods=["POST"])
def analyze_sentiment():
    file_path = request.json.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"msg": "Invalid file path", "success": False})

    _, _, retriever_or_text = process_pdf(file_path)
    conversation_text = get_conversation_text()
    
    result = run_sentiment_analysis(retriever_or_text, conversation_text)
    
    # Check if we have visualization paths in the result
    wordcloud_path = None
    wordcloud_match = re.search(r"<SENTIMENT_WORDCLOUD>(.*?)</SENTIMENT_WORDCLOUD>", result)
    if wordcloud_match:
        wordcloud_path = wordcloud_match.group(1)
        # Remove the tags from the text result
        result = re.sub(r"<SENTIMENT_WORDCLOUD>.*?</SENTIMENT_WORDCLOUD>", "", result)
    
    sentiment_viz_path = None
    sentiment_viz_match = re.search(r"<SENTIMENT_VISUALIZATION>(.*?)</SENTIMENT_VISUALIZATION>", result)
    if sentiment_viz_match:
        sentiment_viz_path = sentiment_viz_match.group(1)
        # Remove the tags from the text result
        result = re.sub(r"<SENTIMENT_VISUALIZATION>.*?</SENTIMENT_VISUALIZATION>", "", result)
    
    # Create a list of all available visualization paths
    visualization_paths = []
    if wordcloud_path:
        visualization_paths.append(wordcloud_path)
    if sentiment_viz_path:
        visualization_paths.append(sentiment_viz_path)
    
    return jsonify({
        "msg": result, 
        "analysis_type": "sentiment", 
        "success": True, 
        "mmr_used": USE_MMR,
        "has_visualization": len(visualization_paths) > 0,
        "visualization_paths": visualization_paths
    })

@app.route('/analyze/additional', methods=["POST"])
def analyze_additional():
    file_path = request.json.get('file_path')
    analysis_type = request.json.get('analysis_type', 'Custom analysis')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"msg": "Invalid file path", "success": False})

    _, _, retriever_or_text = process_pdf(file_path)
    conversation_text = get_conversation_text()
    
    result = run_additional_analysis(retriever_or_text, conversation_text, analysis_type)
    return jsonify({"msg": result, "analysis_type": "additional", "success": True, "mmr_used": USE_MMR})

@app.route('/upload', methods=["POST", "OPTIONS"])
def upload_file():
    if request.method == "OPTIONS":
        return "", 204

    print("Received upload request")
    file = request.files.get('uploaded_file')
    if not file:
        return jsonify({"msg": "No file received", "success": False})

    filename = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    print(f"Saved file: {file_path}")

    if not filename.lower().endswith('.pdf'):
        return jsonify({"msg": "Please upload a PDF file", "success": False})

    # Ensure Ollama is running
    if not ensure_ollama_running():
        return jsonify({"msg": "Analysis service unavailable. Could not start Ollama.", "success": False})

    # Return the file path so the client can request analyses
    return jsonify({
        "msg": "File uploaded successfully. Start sequential analysis.",
        "file_path": file_path,
        "success": True
    })

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/main.js')
def serve_js():
    return app.send_static_file('main.js')

if __name__ == '__main__':
    # Clear the Chroma database to resolve dimension mismatch
    clear_chroma_db()
    
    # Start Ollama if needed
    ensure_ollama_running()
    app.run(host='0.0.0.0', port=5000, debug=True)

