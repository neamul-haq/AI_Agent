from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os, requests, shutil
import pandas as pd

# Step 1: Remove old restaurant DB (optional but clean)
db_location = "./chroma_shopping_db"
if os.path.exists(db_location):
    shutil.rmtree(db_location)

# Step 2: Fetch products from the API
response = requests.get("https://fakestoreapi.in/api/products")
products = response.json()

# Step 3: Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []   
    ids = []
    
    for i, product in enumerate(products):
        content = (
            f"Title: {product['title']}\n"
            f"Description: {product['description']}\n"
            f"Category: {product['category']}\n"
            f"Price: ${product['price']}\n"
            f"Rating: {product.get('rating', {}).get('rate', 'N/A')}"
        )
        document = Document(
            page_content=content,
            metadata={
                "category": product["category"],
                "price": product["price"],
                "api_link": f"https://fakestoreapi.in/api/products/{product['id']}",
            },
            id=str(product["id"]),
        )
        documents.append(document)
        ids.append(str(product["id"]))
    

    # Embed helpful API documentation-like entries
    api_docs = [
        {
            "description": "Get a full list of all products available.",
            "url": "https://fakestoreapi.in/api/products",
        },
        {
            "description": "Get detailed info of a single product by its ID.",
            "url": "https://fakestoreapi.in/api/products/1",
        },
        {
            "description": "Get paginated product results using page numbers.",
            "url": "https://fakestoreapi.in/api/products?page=2",
        },
        {
            "description": "Limit the number of returned products.",
            "url": "https://fakestoreapi.in/api/products?limit=20",
        },
        {
            "description": "View products filtered by category (like mobile, tv, etc).",
            "url": "https://fakestoreapi.in/api/products/category",
        },
        {
            "description": "Get only mobile products.",
            "url": "https://fakestoreapi.in/api/products/category?type=mobile",
        },
        {
            "description": "Get TVs sorted in descending order (e.g., price).",
            "url": "https://fakestoreapi.in/api/products/category?type=tv&sort=desc",
        },
    ]

    for i, doc in enumerate(api_docs, start=len(products)):
        documents.append(
            Document(
                page_content=f"{doc['description']}\nURL: {doc['url']}",
                metadata={"type": "api_doc", "url": doc["url"]},
                id=f"api-{i}"
            )
        )
        ids.append(f"api-{i}")

        
vector_store = Chroma(
    collection_name="shopping_products",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)