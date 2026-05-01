import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Loading API key from secret.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "../05_src/.secrets")
load_dotenv(dotenv_path, override=True)
API_KEY = os.getenv("API_GATEWAY_KEY")

# Creating an OpenAI client.
client = OpenAI(
    base_url = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key = os.getenv("API_GATEWAY_KEY"),
    default_headers = {"x-api-key": os.getenv("API_GATEWAY_KEY")}
)

# Initialize ChromaDB client with persistent storage.
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory=os.path.join(BASE_DIR, "chroma_db")
    )
)

# Creating a variable to for local embeddings.
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Creating a collection for storing embeddings using the specified embedding function.
collection = chroma_client.get_or_create_collection(
    name="semantic_collection",
    embedding_function=embedding_function
)

# Create a functino to Load the dataset into Chroma. Only run once.
def load_data():
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning is a type of machine learning that uses neural networks with many layers.",
        "Natural language processing allows computers to understand human language.",
        "Supervised learning uses labeled data to train models.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Reinforcement learning is based on rewards and penalties.",
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        ids=ids
    )

    print("Data loaded into ChromaDB")


# Perform semantic search using ChromaDB to retrieve relevant documents,
# then use a GPT model (RAG approach) to generate a concise answer
# based on the retrieved context.
def semantic_search(query: str) -> str:
    try:
        results = collection.query(
            query_texts=[query],
            n_results=2
        )

        retrieved_docs = results["documents"][0]

        # Combine retrieved docs
        context = "\n".join(retrieved_docs)

        # Generate final answer using API (RAG style)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using the provided context."
                },
                {
                    "role": "user",
                    "content": f"""
                    Context:
                    {context}

                    Question:
                    {query}

                    Provide a helpful and concise answer.
                    """
                }
            ],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error in semantic search: {str(e)}"


# Creating test samples for testing.
if __name__ == "__main__":
    # Run ONLY once to load data
    # load_data()

    question = "What is deep learning?"
    answer = semantic_search(question)

    print("\nSemantic Search Output:\n")
    print(answer)
