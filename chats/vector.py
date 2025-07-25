import os, requests, shutil
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

db_location = "./chroma_shopping_db"
# if os.path.exists(db_location):
#     shutil.rmtree(db_location)
# Fetch products
response = requests.get("https://fakestoreapi.in/api/products")
data = response.json()
products = data.get("products", [])

# Embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

add_documents = not os.path.exists(db_location)

if add_documents:
    documents, ids = [], []
    # API doc list
    # api_docs = [
    #     {"description": "Get a full list of all products available.", "url": "https://fakestoreapi.in/api/products"},
    #     {"description": "Get detailed info of a single product by its ID.", "url": "https://fakestoreapi.in/api/products/1"},
    #     {"description": "Get paginated product results using page numbers.", "url": "https://fakestoreapi.in/api/products?page=2"},
    #     {"description": "Limit the number of returned products.", "url": "https://fakestoreapi.in/api/products?limit=20"},
    #     {"description": "View products filtered by category (like mobile, tv, etc).", "url": "https://fakestoreapi.in/api/products/category"},
    #     {"description": "Get only mobile products.", "url": "https://fakestoreapi.in/api/products/category?type=mobile"},
    #     {"description": "Get TVs sorted in descending order (e.g., price).", "url": "https://fakestoreapi.in/api/products/category?type=tv&sort=desc"},
    # ]
    
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

    # for i, doc in enumerate(api_docs, start=len(products)):
    #     documents.append(
    #         Document(
    #             page_content=f"{doc['description']}\nURL: {doc['url']}",
    #             metadata={"type": "api_doc", "url": doc["url"]},
    #             id=f"api-{i}"
    #         )
    #     )
    #     ids.append(f"api-{i}")
else:
    print("Database already exists, skipping document addition.")
    
vector_store = Chroma(
    collection_name="shopping_products",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    print(f"Adding {len(documents)} documents...")
    vector_store.add_documents(documents=documents, ids=ids)
else:
    print("Database exists, skipping addition.")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})