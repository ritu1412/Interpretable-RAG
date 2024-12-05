# Interpretable RAG System

[![Streamlit App](https://img.shields.io/badge/Streamlit-Online-orange)](https://interpretable-rag.streamlit.app/)

This project introduces an **interpretable Retrieval-Augmented Generation (RAG) system** with **t-SNE visualizations** to help users better understand how retrieved chunks contribute to the final response generated by a large language model (LLM). The tool provides insights into chunk relevance, allows users to tweak chunking parameters interactively, and avoids time wasted on A/B testing.

## Features

### 1. **t-SNE Visualization**
- Visualizes embeddings of:
  - Chunks
  - Retrieved Chunks
  - Query
  - Response
- Helps determine how close the response is to retrieved chunks, especially when dealing with a large number of chunks.

### 2. **Interactive Chunking**
- Supports multiple chunking techniques:
  - **Fixed Size**: Define chunk size based on word count.
  - **Sentence-Based**: Define chunks based on the number of sentences.
- Allows iterative testing and optimization of chunking parameters to find the best setup for your RAG system.

### 3. **Retrieval-Augmented Generation**
- Efficiently retrieves relevant chunks from a document using embeddings.
- Generates context-aware responses with OpenAI's GPT models.

### 4. **Streamlit Web App**
- Intuitive web-based interface.
- Easily upload PDFs, process them, and interact with the RAG pipeline.

### 5. **Future Enhancements**
- Add support for more interpretable techniques (e.g., **PCA**, **UMAP**).
- Allow users to select LLM model

---

## How It Works

1. **Upload a Document**:
   - Upload a PDF file containing text data.
   - Extracted text is displayed for reference.

2. **Chunk the Text**:
   - Choose a chunking method (Fixed Size or Sentence-Based).
   - Adjust chunking parameters interactively.

3. **Retrieve Relevant Chunks**:
   - Enter a query to retrieve the most relevant chunks.
   - Visualize how the retrieved chunks relate to the query and response using t-SNE.

4. **Generate and Visualize**:
   - Generate a response based on retrieved chunks.
   - Visualize embeddings of all chunks, query, and response with t-SNE.

---

## Key Benefits

- **Avoid A/B Testing**: Quickly identify optimal chunking parameters and embedding configurations without manual experimentation.
- **Improve Interpretability**: Gain insights into the relationship between retrieved chunks and the response.
- **Iterative Workflow**: Easily test different chunking and embedding settings in real-time.

---

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Interpretable-RAG.git
   cd Interpretable-RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Add your API key to `.streamlit/secrets.toml`:
     ```toml
     [api_keys]
     OPENAI_API_KEY = "your-openai-api-key"
     ```

4. Run the app locally:
   ```bash
   streamlit run app.py
   ```

---

## Usage

Visit the hosted Streamlit web app here: [Interpretable RAG](https://interpretable-rag.streamlit.app/).

### Example Workflow

1. Upload a PDF document.
2. Adjust chunking parameters and process the document.
3. Enter a query and retrieve relevant chunks.
4. Generate and visualize the response.
5. Use the t-SNE plot to analyze embedding relationships.

---

## File Structure

```
.
├── app.py              # Main Streamlit app file
├── chunking.py         # Chunking logic for fixed size and sentence-based chunking
├── constants.py        # Constants for embedding models and OpenAI API
├── ocr.py              # PDF text extraction logic
├── retrieval.py        # ChromaDB-based chunk retrieval
├── tsne.py             # t-SNE visualization logic
├── about.md            # Markdown content for the "About" section in the app
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # Streamlit secrets for API keys
```

---

## Future Developments

1. **Model Selection**:
   - Allow users to select and compare different language models and embedding methods.

2. **Additional Visualization Techniques**:
   - Integrate **PCA** and **UMAP** for alternative visualizations of high-dimensional embeddings.

3. **Performance Metrics**:
   - Include tools for evaluating retrieval and generation performance.

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## References and Inspiration

https://projector.tensorflow.org/
https://arxiv.org/html/2402.01761v1
https://arxiv.org/abs/2406.05794
https://arxiv.org/html/2405.13000v1
[Duke explainable AI course](https://pratt.duke.edu/news/xai-coursera/#:~:text=%2C%E2%80%9D%20Bent%20said.-,Explainable%20AI%20and%20interpretable%20ML%20enable%20us%20to%20use%20knowledge,make%20based%20on%20these%20predictions.&text=The%20online%20specialization%20includes%20three,Interpretable%20Machine%20Learning)
