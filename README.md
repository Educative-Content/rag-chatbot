# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LlamaIndex, Chroma, Groq, and Gradio. This application loads documents from a local directory, creates vector embeddings, and provides a conversational interface for querying the documents with context-aware responses.

> Note: This project was created by Educative and serves as the supporting repository for an accompanying blog post.

## Features

- **Document Ingestion**: Automatically loads and processes documents (PDFs, TXT, MD) from the `./data` directory
- **Vector Storage**: Uses ChromaDB for persistent vector storage and retrieval
- **Conversational AI**: Powered by Groq's LLM with conversation history support for follow-up questions
- **Web Interface**: Simple Gradio-based chat UI for easy interaction
- **Clean Rebuild**: Option to rebuild the index from scratch, ensuring clean data state
- **Source Attribution**: Displays relevant source documents for each response

## Installation

1. **Clone or download the project** to your local machine.

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv ragbot
   source ragbot/bin/activate  # On Windows: ragbot\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** in a `.env` file (see Configuration section below).

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Required: Your Groq API key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Customize these as needed
GROQ_MODEL=llama-3.1-8b-instant
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION=rag_collection
DATA_DIR=./data
```

- **GROQ_API_KEY**: Get this from [Groq Console](https://console.groq.com/)
- **GROQ_MODEL**: The LLM model to use (default: llama-3.1-8b-instant)
- **EMBED_MODEL**: The embedding model for document vectors (default: all-MiniLM-L6-v2)
- **TOP_K**: Number of top results to retrieve (default: 4)
- **CHROMA_DIR**: Directory for ChromaDB storage (default: ./chroma_db)
- **CHROMA_COLLECTION**: Chroma collection name (default: rag_collection)
- **DATA_DIR**: Directory containing your documents (default: ./data)

## Usage

1. **Add your documents** to the `./data` directory (supports PDF, TXT, MD files).

2. **Run the application**:
   ```bash
   python main.py
   ```
   This will load existing index if available, or build from scratch if not.

3. **Rebuild index** (if you add new documents):
   ```bash
   python main.py --rebuild
   ```
   This deletes the old index and creates a fresh one.

4. **Access the chat interface**: Open your browser and go to `http://127.0.0.1:7860` (default Gradio URL).

5. **Ask questions**: Type your queries in the chat interface. The bot will respond with context from your documents and cite sources.

## Project Structure

- `main.py`: Main application script
- `data/`: Directory for source documents
- `chroma_db/`: Persistent vector database storage
- `.env`: Environment configuration file (not included in repo)
- `README.md`: This file

## Technologies Used

- **LlamaIndex**: For RAG implementation and document processing
- **ChromaDB**: Vector database for embeddings storage
- **Groq**: LLM provider for conversational responses
- **Gradio**: Web interface for the chat application
- **Sentence Transformers**: For generating document embeddings

## Troubleshooting

- **No API Key**: Ensure `GROQ_API_KEY` is set in `.env` or environment.
- **No Documents**: Add files to `./data` and run with `--rebuild`.
- **Port Issues**: Change Gradio port if 7860 is in use (modify `app.launch()` in `main.py`).
- **Dependencies**: If you encounter import errors, double-check installed packages.

## Contributing

Feel free to fork, modify, and submit pull requests for improvements!

## License

This project is open-source and available under the MIT License.