import os 
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI and ChromaDB clients
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

client = OpenAI(api_key=openai_key)

# Function to load .txt documents from a directory
def load_documents_from_directory(directory_path):
    print("Loading documents...")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

#directory_path=r"C:\Users\prvtg\OneDrive - IRI\Desktop\LLM\news_articles"
#documents=load_documents_from_directory(directory_path)
#print(f"loaded {len(documents)},documents")
#for doc in documents:
#   print(f"{doc['id']}:{len(doc['text'])} characters")


# Function to split documents into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load and split documents
directory_path = r"C:\Users\prvtg\OneDrive - IRI\Desktop\LLM\news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents.")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print(f"{doc['id']}->{len(chunks)}chunks")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})
print("total chunks", len(chunked_documents))

# Function to get embedding from OpenAI
def get_openai_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding

# Generate and store embeddings in ChromaDB
for doc in chunked_documents:
    print(f"Generating embedding for {doc['id']}...")
    doc["embedding"] = get_openai_embedding(doc["text"])
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

# Function to query relevant document chunks
def query_documents(question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    #Below query is to flatten the nested after querying the ChromaDB
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks

# Function to generate response using GPT
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question and answering tasks. Use the following "
        "retrieved context to answer the question. If you don't know the answer, say "
        "you don't know. Use three sentences maximum and keep the answers concise."
        f"\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

# Example usage
question = "Tell me about Databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print("\nAnswer:\n", answer)

