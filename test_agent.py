from shopping_ai_agent import ShoppingAIAgent
from langchain.chat_models import ChatOpenAI 
import os, requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

db_location = "./chroma_shopping_db2"
response = requests.get("https://fakestoreapi.in/api/products")
data = response.json()
products = data.get("products", [])

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

add_documents = not os.path.exists(db_location)

if add_documents:
    documents, ids = [], [] 
    for product in products:
        content = (
            f"Product Title: {product['title']}\n"
            f"Product Price: ${product['price']}"
            f"Product Model: {product.get('model', 'N/A')}\n"
            f"Product Category: {product['category']}\n"
            f"Product Description: {product['description']}\n"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "category": product["category"],
                    "price": product["price"],
                    "api_link": f"https://fakestoreapi.in/api/products/{product['id']}",
                },
                id=str(product["id"]),
            )
        )
        ids.append(str(product["id"]))
    
vector_store = Chroma(
    collection_name="shopping_products",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
# Initialize model
model = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="gemma2-9b-it"
)

# Initialize agent
agent = ShoppingAIAgent(model=model, retriever=retriever)

# Chat session
print("🛍️ Welcome to the AI Shopping Assistant! Type 'exit' to quit.")
session_id = "test-2"

while True:
    user_input = input("\n🧑 You: ")
    if user_input.lower().strip() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    response, issue = agent.generate_response(user_input, session_id=session_id)

    print("🤖 Assistant:", response)
    if issue:
        print("⚠️ Issue Detected:", issue)