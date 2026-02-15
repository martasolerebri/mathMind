import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network
import tempfile
from collections import defaultdict

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Goodreads AI Librarian", page_icon="📖", layout="wide")

# Custom CSS for better book cards
st.markdown("""
<style>
.book-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.book-title {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.book-author {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}
.book-rating {
    font-size: 1.5rem;
}
.stat-box {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📖 Goodreads AI Librarian")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get it from console.groq.com")
    hf_api_key = st.text_input("Hugging Face API Key", type="password", help="For embeddings model")
    
    st.divider()
    
    st.header("Upload Library")
    uploaded_file = st.file_uploader("Upload Goodreads CSV", type="csv")
    
    if uploaded_file:
        st.success("✅ Library loaded!")
    
    st.divider()
    st.markdown("""
    **How to export:**
    1. Go to Goodreads (desktop)
    2. My Books → Tools
    3. Import and export
    4. Export Library
    """)
    
    id_model = "llama-3.3-70b-versatile"
    temperature = 0.5

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

@st.cache_data
def parse_goodreads_csv(file):
    """Parse CSV and return structured book data"""
    df = pd.read_csv(file)
    
    required_cols = ['Title', 'Author']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        return None, None
    
    books = []
    for idx, row in df.iterrows():
        # Get the exclusive shelf (read, to-read, currently-reading)
        exclusive_shelf = str(row.get('Exclusive Shelf', 'read')).lower()
        
        book = {
            'id': idx,
            'title': str(row.get('Title', 'Unknown')),
            'author': str(row.get('Author', 'Unknown')),
            'rating': row.get('My Rating', 0),
            'year_published': row.get('Year Published', ''),
            'date_read': str(row.get('Date Read', '')),
            'date_added': str(row.get('Date Added', '')),
            'exclusive_shelf': exclusive_shelf,
            'shelves': str(row.get('Bookshelves', '')),
            'review': str(row.get('My Review', '')),
            'avg_rating': row.get('Average Rating', 0),
            'num_pages': row.get('Number of Pages', 0),
        }
        
        # Build text for embedding (focus on themes/content)
        text_parts = [f"{book['title']} by {book['author']}"]
        if book['shelves'] and book['shelves'] != 'nan':
            text_parts.append(f"Genres/Tags: {book['shelves']}")
        if book['review'] and book['review'] != 'nan' and len(book['review']) > 10:
            text_parts.append(f"Review: {book['review'][:500]}")
        
        book['embedding_text'] = ". ".join(text_parts)
        books.append(book)
    
    return books, df

def process_csv_to_retriever(books, embeddings_model):
    """Turn books into searchable documents for RAG"""
    documents = []
    for book in books:
        content = f"Title: {book['title']}\nAuthor: {book['author']}\n"
        content += f"My Rating: {book['rating']}/5\n"
        content += f"Status: {book['exclusive_shelf']}\n"
        content += f"Shelves/Tags: {book['shelves']}\n"
        if book['review'] and book['review'] != 'nan':
            content += f"My Review: {book['review']}\n"
            
        doc = Document(page_content=content, metadata={"title": book['title'], "author": book['author']})
        documents.append(doc)
        
    vectorstore = FAISS.from_documents(documents, embedding=embeddings_model)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

@st.cache_data
def generate_embeddings(_embeddings_model, books):
    """Generate embeddings for thematic similarity"""
    texts = [book['embedding_text'] for book in books]
    with st.spinner("Generating embeddings for graph analysis..."):
        book_embeddings = _embeddings_model.embed_documents(texts)
    return np.array(book_embeddings)

def extract_genres_and_themes(book, _llm):
    """Use LLM to extract clean genres/themes from book info"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a book categorization expert. Extract 2-4 primary genres or themes."),
        ("human", """Analyze this book and return ONLY a JSON list of 2-4 primary genres/themes:

Title: {title}
Author: {author}
Shelves/Tags: {shelves}
Review snippet: {review}

Return format: {{"themes": ["theme1", "theme2", "theme3"]}}
Themes should be specific literary genres or concepts (e.g., "dystopian fiction", "feminist literature", "space opera", "coming of age")
""")
    ])
    
    chain = prompt | _llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "title": book['title'],
            "author": book['author'],
            "shelves": book['shelves'],
            "review": book['review'][:200] if book['review'] != 'nan' else ""
        })
        
        # Clean JSON response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        
        result = json.loads(response)
        return result.get('themes', [])
    except Exception as e:
        # Fallback to shelf tags
        if book['shelves'] and book['shelves'] != 'nan':
            return [tag.strip() for tag in book['shelves'].split() if tag.strip()][:3]
        return ["general"]

def build_thematic_graph(books, embeddings_matrix, _llm, similarity_threshold=0.35, max_books=50):
    """Build graph based on shared themes and embedding similarity"""
    
    # Limit books for performance
    if len(books) > max_books:
        st.info(f"Analyzing top {max_books} books for graph (sorted by rating)")
        # Prioritize highly rated books
        sorted_books = sorted(books, key=lambda x: (x['rating'], x['avg_rating']), reverse=True)
        books = sorted_books[:max_books]
        embeddings_matrix = embeddings_matrix[[b['id'] for b in books]]
    
    G = nx.Graph()
    
    # Extract themes for each book
    st.write("🔍 Extracting themes from books...")
    progress_bar = st.progress(0)
    
    book_themes = {}
    for i, book in enumerate(books):
        themes = extract_genres_and_themes(book, _llm)
        book_themes[book['id']] = themes
        
        # Add node
        G.add_node(
            book['id'],
            title=book['title'],
            author=book['author'],
            rating=book.get('rating', 0),
            themes=themes,
            label=book['title'][:40] + "..." if len(book['title']) > 40 else book['title']
        )
        
        progress_bar.progress((i + 1) / len(books))
    
    progress_bar.empty()
    
    # Build theme index for faster lookup
    theme_to_books = defaultdict(list)
    for book_id, themes in book_themes.items():
        for theme in themes:
            theme_to_books[theme.lower()].append(book_id)
    
    # Add edges based on shared themes
    st.write("🔗 Connecting books with shared themes...")
    edges_added = set()
    
    for book in books:
        book_id = book['id']
        book_idx = books.index(book)
        
        # Find books with shared themes
        connected_books = set()
        for theme in book_themes[book_id]:
            connected_books.update(theme_to_books[theme.lower()])
        
        # Remove self
        connected_books.discard(book_id)
        
        # For each potential connection, check embedding similarity
        for other_id in connected_books:
            edge_key = tuple(sorted([book_id, other_id]))
            if edge_key in edges_added:
                continue
            
            other_idx = next(i for i, b in enumerate(books) if b['id'] == other_id)
            
            # Calculate similarity
            similarity = cosine_similarity(
                embeddings_matrix[book_idx].reshape(1, -1),
                embeddings_matrix[other_idx].reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                # Find shared themes
                shared_themes = set(book_themes[book_id]) & set(book_themes[other_id])
                
                if shared_themes:
                    G.add_edge(
                        book_id,
                        other_id,
                        weight=float(similarity),
                        themes=", ".join(shared_themes),
                        title=f"Shared themes: {', '.join(shared_themes)}\nSimilarity: {similarity:.2f}"
                    )
                    edges_added.add(edge_key)
    
    st.success(f"✅ Graph built: {len(G.nodes)} books, {len(G.edges)} thematic connections")
    return G, book_themes

def visualize_graph(G, books, book_themes):
    """Create interactive visualization"""
    
    net = Network(height="700px", width="100%", bgcolor="#1e1e1e", font_color="white")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=250, spring_strength=0.001)
    
    # Color by primary theme
    theme_colors = {
        'fantasy': '#9b59b6',
        'sci-fi': '#3498db',
        'science fiction': '#3498db',
        'romance': '#e74c3c',
        'mystery': '#34495e',
        'thriller': '#c0392b',
        'historical': '#d35400',
        'dystopian': '#7f8c8d',
        'literary fiction': '#27ae60',
        'classic': '#f39c12',
        'young adult': '#e67e22',
        'horror': '#8e44ad',
        'non-fiction': '#16a085',
        'biography': '#2ecc71',
    }
    
    def get_node_color(themes, rating):
        # Try to match theme to color
        for theme in themes:
            theme_lower = theme.lower()
            for key, color in theme_colors.items():
                if key in theme_lower:
                    return color
        
        # Fallback to rating-based color
        if rating >= 4:
            return "#2ecc71"  # Green
        elif rating >= 3:
            return "#f39c12"  # Orange
        elif rating > 0:
            return "#e74c3c"  # Red
        else:
            return "#95a5a6"  # Gray
    
    # Add nodes
    for node in G.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        book = next(b for b in books if b['id'] == node_id)
        themes = book_themes.get(node_id, [])
        
        net.add_node(
            node_id,
            label=node_data['label'],
            title=f"<b>{node_data['title']}</b><br>{node_data['author']}<br>Rating: {node_data['rating']}⭐<br>Themes: {', '.join(themes)}",
            color=get_node_color(themes, node_data['rating']),
            size=15 + (node_data['rating'] * 4)
        )
    
    # Add edges
    for edge in G.edges(data=True):
        net.add_edge(
            edge[0],
            edge[1],
            title=edge[2].get('title', ''),
            value=edge[2].get('weight', 0.5) * 3,
            color={'color': '#636363', 'opacity': 0.4}
        )
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as file:
            html_content = file.read()
    
    return html_content

# Main App
if uploaded_file:
    # Parse CSV
    if 'books' not in st.session_state or 'df' not in st.session_state:
        books, df = parse_goodreads_csv(uploaded_file)
        if books is None:
            st.stop()
        
        st.session_state.books = books
        st.session_state.df = df
    
    books = st.session_state.books
    df = st.session_state.df
    
    # Create retriever for chat
    if 'retriever' not in st.session_state:
        with st.spinner("Processing library for chat..."):
            st.session_state.retriever = process_csv_to_retriever(books, embeddings)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Library", "🕸️ Connection Graph"])
    
    # TAB 1: CHAT
    with tab1:
        st.markdown("### Chat with your library")
        st.markdown("*Ask questions like: 'What 5-star fantasy books have I read?' or 'Recommend me something from my to-read list'*")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask about your books..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            system_prompt = (
                "You are a friendly, knowledgeable AI librarian analyzing a user's Goodreads library. "
                "Use the following retrieved context (which contains books the user has shelved, their ratings, and their reviews) "
                "to answer their question or provide recommendations. "
                "If they ask for recommendations, prioritize books from their own library context. "
                "Keep your responses conversational, insightful, and concise. "
                "If you don't know the answer, just say so.\n\n"
                "Context: {context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            chain = (
                {"context": st.session_state.retriever, "input": RunnablePassthrough()}
                | qa_prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                with st.spinner("Scanning the shelves..."):
                    response = chain.invoke(prompt)
                    clean_response = response.split("</think>")[-1].strip() if "</think>" in response else response
                    
                    st.write(clean_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
    
    # TAB 2: LIBRARY VIEW
    with tab2:
        st.markdown("### 📖 Your Library at a Glance")
        
        # Statistics
        read_books = [b for b in books if b['exclusive_shelf'] == 'read']
        to_read_books = [b for b in books if b['exclusive_shelf'] == 'to-read']
        currently_reading = [b for b in books if b['exclusive_shelf'] == 'currently-reading']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Books", len(books))
        with col2:
            st.metric("✅ Read", len(read_books))
        with col3:
            st.metric("📖 Currently Reading", len(currently_reading))
        with col4:
            st.metric("🔖 To Read", len(to_read_books))
        
        st.divider()
        
        # Read Books Section
        if read_books:
            st.subheader(f"✅ Books You've Read ({len(read_books)})")
            
            # Sort options
            sort_by = st.selectbox("Sort by:", ["Rating (High to Low)", "Rating (Low to High)", "Title", "Author", "Date Read (Recent)"])
            
            if sort_by == "Rating (High to Low)":
                read_books = sorted(read_books, key=lambda x: x['rating'], reverse=True)
            elif sort_by == "Rating (Low to High)":
                read_books = sorted(read_books, key=lambda x: x['rating'])
            elif sort_by == "Title":
                read_books = sorted(read_books, key=lambda x: x['title'])
            elif sort_by == "Author":
                read_books = sorted(read_books, key=lambda x: x['author'])
            elif sort_by == "Date Read (Recent)":
                read_books = sorted(read_books, key=lambda x: x['date_read'], reverse=True)
            
            # Display in grid
            cols = st.columns(3)
            for i, book in enumerate(read_books):
                with cols[i % 3]:
                    stars = "⭐" * int(book['rating']) if book['rating'] > 0 else "Not rated"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem; color: white;">
                            <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem;">
                                {book['title'][:50]}{"..." if len(book['title']) > 50 else ""}
                            </div>
                            <div style="opacity: 0.9; margin-bottom: 0.5rem;">
                                by {book['author']}
                            </div>
                            <div style="font-size: 1.2rem;">
                                {stars}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.divider()
        
        # To-Read Section
        if to_read_books:
            st.subheader(f"🔖 Your To-Read List ({len(to_read_books)})")
            
            cols = st.columns(3)
            for i, book in enumerate(to_read_books[:30]):  # Limit to 30 for performance
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem; color: white;">
                            <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem;">
                                {book['title'][:50]}{"..." if len(book['title']) > 50 else ""}
                            </div>
                            <div style="opacity: 0.9;">
                                by {book['author']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if len(to_read_books) > 30:
                st.info(f"Showing 30 of {len(to_read_books)} to-read books")
    
    # TAB 3: CONNECTION GRAPH
    with tab3:
        st.markdown("### 🕸️ Thematic Connection Map")
        st.markdown("*Discover how your books connect through shared genres and themes*")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            max_books_graph = st.slider("Books to analyze", 20, 100, 40, help="More books = slower")
            similarity_threshold = st.slider("Connection threshold", 0.2, 0.6, 0.35, 0.05)
        
        if st.button("🔍 Generate Connection Graph", type="primary"):
            # Generate embeddings if not cached
            if 'embeddings_matrix' not in st.session_state:
                embeddings_matrix = generate_embeddings(embeddings, books)
                st.session_state.embeddings_matrix = embeddings_matrix
            
            embeddings_matrix = st.session_state.embeddings_matrix
            
            # Build graph
            G, book_themes = build_thematic_graph(
                books,
                embeddings_matrix,
                llm,
                similarity_threshold=similarity_threshold,
                max_books=max_books_graph
            )
            
            st.session_state.graph = G
            st.session_state.book_themes = book_themes
        
        # Display graph
        if 'graph' in st.session_state:
            st.divider()
            
            # Filter to only books in graph
            graph_books = [b for b in books if b['id'] in st.session_state.graph.nodes()]
            
            html_content = visualize_graph(
                st.session_state.graph,
                graph_books,
                st.session_state.book_themes
            )
            
            st.components.v1.html(html_content, height=750, scrolling=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Books in Graph", len(st.session_state.graph.nodes))
            with col2:
                st.metric("Thematic Connections", len(st.session_state.graph.edges))
            with col3:
                avg_degree = sum(dict(st.session_state.graph.degree()).values()) / len(st.session_state.graph.nodes)
                st.metric("Avg Connections/Book", f"{avg_degree:.1f}")
            
            # Theme distribution
            st.divider()
            st.subheader("📊 Most Common Themes in Your Library")
            
            all_themes = []
            for themes in st.session_state.book_themes.values():
                all_themes.extend(themes)
            
            from collections import Counter
            theme_counts = Counter(all_themes).most_common(10)
            
            col1, col2 = st.columns(2)
            for i, (theme, count) in enumerate(theme_counts):
                with col1 if i % 2 == 0 else col2:
                    st.markdown(f"**{theme.title()}**: {count} books")

else:
    st.info("👆 Upload your Goodreads library CSV from the sidebar to get started!")
    
    st.markdown("""
    ### Features:
    
    **💬 Chat Tab**: Ask questions about your reading history
    - "What 5-star books have I read?"
    - "Recommend me a sci-fi from my to-read list"
    - "What fantasy books did I rate highly?"
    
    **📚 Library Tab**: Browse your collection visually
    - See all your read books sorted by rating
    - Browse your to-read list
    - Quick stats about your reading habits
    
    **🕸️ Graph Tab**: Discover thematic connections
    - AI-powered theme extraction
    - Interactive network showing how books relate
    - Color-coded by genre and rating
    - Hover over connections to see shared themes
    """)