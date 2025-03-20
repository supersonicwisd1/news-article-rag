# News Article Chatbot

A powerful RAG (Retrieval-Augmented Generation) chatbot that can answer questions about news articles using OpenAI's GPT models and ChromaDB for vector storage.

## Features

- Document loading and processing from a specified directory
- Text chunking with configurable size and overlap
- Vector embeddings using OpenAI's text-embedding-3-small model
- Persistent vector storage using ChromaDB
- Semantic search for relevant document chunks
- Question answering using GPT-3.5-turbo

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd news-article-chat-bot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
news-article-chat-bot/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── chroma_db/         # ChromaDB persistent storage
└── news_articles/     # Directory containing news articles
```

## Usage

1. Place your news articles in the `news_articles/` directory.

2. Run the application:
```bash
python app.py
```

The application will:
- Load documents from the news_articles directory
- Split them into chunks
- Generate embeddings
- Store them in ChromaDB
- Process queries and generate answers

## Key Functions

### Document Processing
- `load_documents(directory)`: Loads text files from the specified directory
- `split_text(text, chunk_size=1000, chunk_overlap=200)`: Splits documents into manageable chunks

### Vector Operations
- `get_openai_embeddings(text)`: Generates embeddings using OpenAI's API
- `query_documents(question, n_results=2)`: Retrieves relevant document chunks for a query

### Answer Generation
- `generate_answer(question, relevant_chunks)`: Generates answers using GPT-3.5-turbo

## Configuration

You can modify the following parameters in `app.py`:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `n_results`: Number of relevant chunks to retrieve (default: 2)

## Dependencies

- openai
- chromadb
- python-dotenv