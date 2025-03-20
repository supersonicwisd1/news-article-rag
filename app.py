import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Creating the embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

# Initialize the Chroma client with persistance
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection_name = "document_qa_collection"

# Creating the collection
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef,
)

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of the moon?"},
    ],
)

print(response.choices[0].message.content)

# Function to load documents from a directory
def load_documents(directory):
    documents = []
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r") as f:
            documents.append(f.read())
    return documents

# Function to split documents into chunks
def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# Load documents from the documents directory
directory_path = "/news_articles"
documents = load_documents(directory_path)

print(f"Loaded {len(documents)} documents from {directory_path}")

# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc)
    print("Spliting docs into chunks")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({
            "id": f"{doc['id']}-{i}",
            "text": chunk,
        })

# print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")

# Embed document
def get_openai_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


for doc in chunked_documents:
    doc["embedding"] = get_openai_embeddings(doc["text"])

# Add documents to the collection
for doc in chunked_documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]],
    )

def query_documents(question, n_results=2):
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    relevant_chunks = [chunk["text"] for chunk in results["documents"][0]]
    return relevant_chunks

def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are a helpful assistant that can answer questions about the following text: "
        f"\n\n<context>{context}</context>\n\n"
        f"Question: {question}\n\n"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}, {"role": "user", "content": question}],
    )
    return response.choices[0].message.content

question = "tell me about AI replacing TV writers?"
results = query_documents(question)
answer = generate_answer(question, results)
print(answer)