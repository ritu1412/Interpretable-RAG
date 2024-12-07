__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import nltk
nltk.download('punkt')

import streamlit as st
from ocr import extract_text_from_pdf
from chunking import chunk_text_fixed_size, chunk_text_by_sentence
from retrieval import index_chunks, retrieve_similar_chunks
from tsne import get_embeddings, plot_tsne
from constants import CHROMADB_DIR, OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS
import openai
import os
import chromadb

openai.api_key = OPENAI_API_KEY


def render_about_section():
    """Render the About section."""
    st.title("About This App")
    st.subheader("Interpretable Retrieval-Augmented Generation (RAG) Web App")
    st.write("This web app is designed to process text data from uploaded PDF files, chunk it, and use state-of-the-art AI models for retrieval and response generation interpretable.")
    markdown_file_path = "about.md"  # Path to your Markdown file
    if os.path.exists(markdown_file_path):
        with open(markdown_file_path, "r") as file:
            about_content = file.read()
        st.markdown(about_content, unsafe_allow_html=True)
    else:
        st.error("The About file (about.md) is missing!")


def render_tool_section():
    """Render the Tool section."""
    st.title("Interpretable RAG Web App")

    # Embedding model selection
    st.sidebar.title("Embedding Settings")
    embedding_model = st.sidebar.selectbox("Select Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])
    embedding_dimensions = st.sidebar.number_input("Embedding Dimensions (optional)", min_value=0, max_value=3072, value=0)
    if embedding_dimensions == 0:
        embedding_dimensions = None

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        #clear the sqlite database
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        client.reset()
        # Save uploaded file to disk
        pdf_path = 'uploaded.pdf'
        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_pdf(pdf_path)
        st.subheader("Extracted Text")
        with st.expander("Show Extracted Text"):
            st.write(text)

        # Chunking options
        st.subheader("Chunking Options")
        chunking_option = st.selectbox("Select Chunking Technique", ["Fixed Size", "By Sentence"])
        if chunking_option == "Fixed Size":
            chunk_size = st.number_input("Chunk Size (words)", min_value=100, max_value=1000, value=500)
            chunks = chunk_text_fixed_size(text, chunk_size=chunk_size)
        elif chunking_option == "By Sentence":
            sentences_per_chunk = st.number_input("Sentences per Chunk", min_value=1, max_value=20, value=5)
            chunks = chunk_text_by_sentence(text, sentences_per_chunk=sentences_per_chunk)

        st.write(f"Number of Chunks: {len(chunks)}")

        # Index chunks
        with st.spinner('Indexing chunks...'):
            index_chunks(chunks)

        # Query input
        st.subheader("Query")
        query = st.text_input("Enter your query")
        if query:
            # Retrieve similar chunks
            with st.spinner('Retrieving similar chunks...'):
                retrieved_chunks, retrieved_indices = retrieve_similar_chunks(query)
            st.subheader("Retrieved Chunks")
            for i, chunk in enumerate(retrieved_chunks):
                with st.expander(f"Chunk {retrieved_indices[i]}"):
                    st.write(chunk)

            # Generate response using OpenAI ChatCompletion API
            with st.spinner('Generating response...'):
                # Prepare the context by combining the retrieved chunks
                context = "\n\n".join(retrieved_chunks)
                # Create the prompt
                prompt = f"Answer the following question based on the context provided:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
            st.subheader("Response")
            st.write(answer)

            # Get embeddings
            with st.spinner('Generating embeddings and plotting t-SNE...'):
                embeddings = get_embeddings(chunks + [query, answer], model=embedding_model, dimensions=embedding_dimensions)
                # Indices
                num_chunks = len(chunks)
                query_index = num_chunks  # Index of the query in embeddings
                response_index = num_chunks + 1  # Index of the response in embeddings
                # Update highlighted indices
                retrieved_indices = [int(idx) for idx in retrieved_indices]
                highlighted_indices = retrieved_indices + [query_index, response_index]
                # Plot t-SNE
                fig = plot_tsne(embeddings, highlighted_indices, num_chunks, query_index, response_index, pre_reduce_dim=True)
                st.subheader("t-SNE Plot")
                st.plotly_chart(fig, use_container_width=True)

        # Clean up uploaded file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    

def main():
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["About", "Tool"])

    if section == "About":
        render_about_section()
    elif section == "Tool":
        render_tool_section()


if __name__ == "__main__":
    main()
