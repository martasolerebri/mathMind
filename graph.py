import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import tempfile

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Goodreads Library Graph", page_icon="📚", layout="wide")
st.title("📚 Goodreads Library Connection Graph")

st.markdown("""
Upload your Goodreads library CSV to visualize thematic connections between your books.
**How to get your CSV:** Go to Goodreads → My Books → Import and export → Export Library
""")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get it from console.groq.com")
    hf_api_key = st.text_input("Hugging Face API Key", type="password", help="For embeddings model")
    
    st.divider()
    
    st.header("Upload Library")
    uploaded_file = st.file_uploader("Upload Goodreads CSV", type="csv")
    
    st.divider()
    
    st.header("Graph Settings")
    max_books = st.slider("Max books to analyze", 10, 100, 30, help="More books = slower processing")
    connection_threshold = st.slider("Connection strength threshold", 0.0, 1.0, 0.3, 0.05, 
                                     help="Higher = fewer connections shown")
    max_connections_per_book = st.slider("Max connections per book", 1, 10, 3)
    
    st.divider()
    id_model = "llama-3.3-70b-versatile"
    temperature = 0.3

if not groq_api_key or not hf_api_key:
    st.warning("Please enter both API Keys (Groq and Hugging Face) in the sidebar to begin.")
    st.stop()

@st.cache_resource
def load_models(groq_key, hf_key):
    llm = ChatGroq(api_key=groq_key, model=id_model, temperature=temperature)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    return llm, embeddings

llm, embeddings = load_models(groq_api_key, hf_api_key)

def parse_goodreads_csv(file):
    """Parse Goodreads export CSV and extract relevant book data"""
    df = pd.read_csv(file)
    
    # Key columns in Goodreads export
    required_cols = ['Title', 'Author']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        return None
    
    # Create clean book entries
    books = []
    for idx, row in df.iterrows():
        book = {
            'id': idx,
            'title': str(row.get('Title', '')),
            'author': str(row.get('Author', '')),
            'rating': row.get('My Rating', 0),
            'year_published': row.get('Year Published', ''),
            'date_read': row.get('Date Read', ''),
            'shelves': str(row.get('Bookshelves', '')),
            'description': str(row.get('My Review', ''))  # Use review as description if available
        }
        
        # Create a text representation for embedding
        book['text_for_embedding'] = f"{book['title']} by {book['author']}. Shelves: {book['shelves']}"
        if book['description'] and book['description'] != 'nan':
            book['text_for_embedding'] += f" Review: {book['description']}"
        
        books.append(book)
    
    return books

@st.cache_data
def generate_embeddings(_embeddings_model, books):
    """Generate embeddings for all books"""
    texts = [book['text_for_embedding'] for book in books]
    
    with st.spinner("Generating embeddings..."):
        book_embeddings = _embeddings_model.embed_documents(texts)
    
    return np.array(book_embeddings)

def find_similar_books(book_idx, embeddings_matrix, top_k=5):
    """Find most similar books using cosine similarity"""
    book_embedding = embeddings_matrix[book_idx].reshape(1, -1)
    similarities = cosine_similarity(book_embedding, embeddings_matrix)[0]
    
    # Get top k similar (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    similar_scores = similarities[similar_indices]
    
    return list(zip(similar_indices, similar_scores))

def extract_themes_with_llm(book1, book2, _llm):
    """Use LLM to identify thematic connections between two books"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a literary analyst. Identify thematic connections between books concisely."),
        ("human", """Compare these two books and identify 1-2 thematic connections:

Book 1: "{title1}" by {author1}
Shelves/Tags: {shelves1}

Book 2: "{title2}" by {author2}
Shelves/Tags: {shelves2}

Return ONLY a JSON object with this format (no other text):
{{"themes": ["theme1", "theme2"], "strength": 0.8}}

Where themes are short phrases (2-4 words) and strength is 0.0-1.0""")
    ])
    
    chain = prompt | _llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "title1": book1['title'],
            "author1": book1['author'],
            "shelves1": book1['shelves'],
            "title2": book2['title'],
            "author2": book2['author'],
            "shelves2": book2['shelves']
        })
        
        # Parse JSON from response
        # Sometimes LLM adds markdown code blocks
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        result = json.loads(response)
        return result.get('themes', []), result.get('strength', 0.5)
    except Exception as e:
        st.warning(f"LLM parsing error: {e}")
        # Fallback: use shelf tags as themes
        common_shelves = set(book1['shelves'].split()) & set(book2['shelves'].split())
        return list(common_shelves)[:2], 0.4

def build_graph(books, embeddings_matrix, threshold, max_connections, _llm):
    """Build network graph of book connections"""
    
    G = nx.Graph()
    
    # Add nodes
    for book in books:
        G.add_node(
            book['id'],
            title=book['title'],
            author=book['author'],
            rating=book.get('rating', 0),
            label=f"{book['title']}\n{book['author']}"
        )
    
    progress_bar = st.progress(0)
    total_books = len(books)
    
    # Add edges based on similarity
    connections_added = 0
    for i, book in enumerate(books):
        similar_books = find_similar_books(i, embeddings_matrix, top_k=max_connections)
        
        for similar_idx, similarity_score in similar_books:
            if similarity_score >= threshold:
                # Use LLM to get thematic connection
                themes, llm_strength = extract_themes_with_llm(
                    books[i], 
                    books[similar_idx], 
                    _llm
                )
                
                # Combine embedding similarity and LLM strength
                final_weight = (similarity_score + llm_strength) / 2
                
                if final_weight >= threshold:
                    G.add_edge(
                        books[i]['id'],
                        books[similar_idx]['id'],
                        weight=final_weight,
                        themes=", ".join(themes) if themes else "Similar themes",
                        title=f"Themes: {', '.join(themes)}\nStrength: {final_weight:.2f}"
                    )
                    connections_added += 1
        
        progress_bar.progress((i + 1) / total_books)
    
    progress_bar.empty()
    st.success(f"Graph built with {len(G.nodes)} books and {len(G.edges)} connections!")
    
    return G

def visualize_graph(G, books):
    """Create interactive visualization using pyvis"""
    
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)
    
    # Color palette for ratings
    def get_color_by_rating(rating):
        if rating >= 4:
            return "#4CAF50"  # Green for high ratings
        elif rating >= 3:
            return "#FFC107"  # Yellow for medium
        elif rating > 0:
            return "#FF5722"  # Red for low
        else:
            return "#9E9E9E"  # Gray for unrated
    
    # Add nodes with styling
    for node in G.nodes():
        book = next(b for b in books if b['id'] == node)
        
        net.add_node(
            node,
            label=book['title'][:30] + "..." if len(book['title']) > 30 else book['title'],
            title=f"<b>{book['title']}</b><br>{book['author']}<br>Rating: {book.get('rating', 'N/A')}⭐",
            color=get_color_by_rating(book.get('rating', 0)),
            size=20 + (book.get('rating', 0) * 5)
        )
    
    # Add edges with styling
    for edge in G.edges(data=True):
        net.add_edge(
            edge[0],
            edge[1],
            title=edge[2].get('title', ''),
            value=edge[2].get('weight', 0.5) * 5,  # Scale for visibility
            color={'color': '#848484', 'opacity': 0.5}
        )
    
    # Save and display
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as file:
            html_content = file.read()
    
    return html_content

# Main app flow
if uploaded_file:
    # Parse CSV
    if 'books' not in st.session_state:
        with st.spinner("Parsing Goodreads CSV..."):
            books = parse_goodreads_csv(uploaded_file)
            if books is None:
                st.stop()
            
            # Limit number of books
            if len(books) > max_books:
                st.info(f"Analyzing top {max_books} books (adjust in sidebar)")
                books = books[:max_books]
            
            st.session_state.books = books
            st.session_state.book_count = len(books)
    
    books = st.session_state.books
    
    # Generate embeddings
    if 'embeddings_matrix' not in st.session_state:
        embeddings_matrix = generate_embeddings(embeddings, books)
        st.session_state.embeddings_matrix = embeddings_matrix
    
    embeddings_matrix = st.session_state.embeddings_matrix
    
    # Build graph
    if st.button("🔍 Analyze Connections", type="primary"):
        with st.spinner("Building graph with LLM analysis..."):
            G = build_graph(
                books, 
                embeddings_matrix, 
                connection_threshold, 
                max_connections_per_book,
                llm
            )
            st.session_state.graph = G
    
    # Display graph
    if 'graph' in st.session_state:
        st.divider()
        st.subheader("📊 Your Library Connection Graph")
        st.markdown("**Green** = High-rated books | **Yellow** = Medium | **Red** = Low-rated | **Gray** = Unrated")
        
        html_content = visualize_graph(st.session_state.graph, books)
        st.components.v1.html(html_content, height=800, scrolling=True)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Books", len(st.session_state.graph.nodes))
        with col2:
            st.metric("Total Connections", len(st.session_state.graph.edges))
        with col3:
            avg_connections = len(st.session_state.graph.edges) / len(st.session_state.graph.nodes) * 2
            st.metric("Avg Connections per Book", f"{avg_connections:.1f}")
        
        # Show sample connections
        st.divider()
        st.subheader("🔗 Sample Thematic Connections")
        
        edges_with_themes = [(u, v, d) for u, v, d in st.session_state.graph.edges(data=True)]
        if edges_with_themes:
            # Show top 5 strongest connections
            edges_with_themes.sort(key=lambda x: x[2]['weight'], reverse=True)
            
            for u, v, data in edges_with_themes[:5]:
                book1 = next(b for b in books if b['id'] == u)
                book2 = next(b for b in books if b['id'] == v)
                
                with st.expander(f"{book1['title']} ↔️ {book2['title']}"):
                    st.write(f"**Themes:** {data['themes']}")
                    st.write(f"**Connection Strength:** {data['weight']:.2f}")
                    st.write(f"**{book1['title']}** by {book1['author']}")
                    st.write(f"**{book2['title']}** by {book2['author']}")

else:
    st.info("👆 Upload your Goodreads library CSV from the sidebar to get started!")
    
    st.markdown("""
    ### How to Export Your Goodreads Library:
    1. Go to [Goodreads.com](https://www.goodreads.com)
    2. Click "My Books"
    3. Scroll down and click "Import and export"
    4. Click "Export Library"
    5. Upload the downloaded CSV here
    
    ### What This App Does:
    - 🔍 Uses **HuggingFace embeddings** to find semantically similar books
    - 🤖 Uses **Groq LLM** to identify specific thematic connections
    - 📊 Creates an **interactive graph** where you can explore your reading patterns
    - 🎨 Color-codes books by your ratings
    - 🔗 Shows thematic connections between books (hover over connections to see themes)
    """)