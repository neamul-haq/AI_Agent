# Shopping AI Agent

A powerful AI-powered shopping assistant built with LangChain that helps customers with product recommendations, queries, and shopping assistance. The agent uses vector embeddings for intelligent product retrieval and maintains conversation memory for personalized interactions.

## Features

- **Intelligent Product Search**: Uses vector embeddings to find relevant products based on user queries
- **Conversation Memory**: Maintains context across chat sessions using summary-based memory
- **Issue Detection**: Optional issue detection and monitoring capabilities
- **Multi-Model Support**: Compatible with various LLM providers (OpenAI, Groq, etc.)
- **Session Management**: Handles multiple user sessions simultaneously
- **Product Database**: Integrates with product APIs and creates searchable vector databases

## Installation

```bash
pip install shopping-ai-agent
```

### Dependencies

The package requires the following dependencies:

```bash
pip install langchain langchain-chroma langchain-ollama langchain-core python-dotenv requests
```

## Quick Start

### Basic Usage

```python
from shopping_ai_agent import ShoppingAIAgent
from langchain.chat_models import ChatOpenAI
import os

# Initialize your LLM model
model = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="gemma2-9b-it"
)

# Set up your product retriever (vector store)
# ... (vector store setup code)

# Initialize the agent
agent = ShoppingAIAgent(model=model, retriever=retriever)

# Generate responses
session_id = "user-123"
response, issue = agent.generate_response("Show me laptops under $1000", session_id)
print(response)
```

### Complete Example with Product Database

```python
from shopping_ai_agent import ShoppingAIAgent
from langchain.chat_models import ChatOpenAI 
import os, requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Database setup
db_location = "./chroma_shopping_db"
response = requests.get("https://fakestoreapi.in/api/products")
data = response.json()
products = data.get("products", [])

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Create vector store
add_documents = not os.path.exists(db_location)

if add_documents:
    documents, ids = [], [] 
    for product in products:
        content = (
            f"Product Title: {product['title']}\n"
            f"Product Price: ${product['price']}\n"
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

# Interactive chat session
print("🛍️ Welcome to the AI Shopping Assistant! Type 'exit' to quit.")
session_id = "test-session"

while True:
    user_input = input("\n🧑 You: ")
    if user_input.lower().strip() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    response, issue = agent.generate_response(user_input, session_id=session_id)

    print("🤖 Assistant:", response)
    if issue:
        print("⚠️ Issue Detected:", issue)
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # if using OpenAI
```

### Model Configuration

The agent supports various LLM providers:

```python
# Groq (recommended for speed)
model = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="gemma2-9b-it"
)

# OpenAI
model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)
```

## API Reference

### ShoppingAIAgent

The main agent class for handling shopping-related conversations.

#### Constructor

```python
ShoppingAIAgent(model, retriever, issue_checker=None)
```

**Parameters:**
- `model`: LangChain LLM model instance
- `retriever`: Vector store retriever for product search
- `issue_checker` (optional): Issue detection model

#### Methods

##### `generate_response(user_input, session_id)`

Generates a response to user input with conversation memory.

**Parameters:**
- `user_input` (str): User's message or query
- `session_id` (str): Unique identifier for the conversation session

**Returns:**
- `tuple`: (response_text, issue) where issue is None if no issues detected

## Memory Management

The agent uses `ConversationSummaryMemory` to maintain context across conversations while keeping memory usage efficient. Each session is tracked separately using session IDs.

```python
# Different users/sessions
agent.generate_response("Show me phones", session_id="user_1")
agent.generate_response("What about tablets?", session_id="user_2")
```

## Vector Store Setup

The agent uses Chroma vector database for efficient product search:

1. **Embeddings**: Uses Ollama embeddings (`mxbai-embed-large`) for text vectorization
2. **Document Structure**: Products are stored as documents with metadata
3. **Retrieval**: Top-k similarity search (default k=5)

## Issue Detection

Optional issue detection can be enabled by providing an `issue_checker` model:

```python
# Initialize with issue detection
agent = ShoppingAIAgent(
    model=model, 
    retriever=retriever, 
    issue_checker=issue_detection_model
)

response, issue = agent.generate_response("I hate this product", session_id)
if issue:
    print(f"Issue detected: {issue}")
```

## Examples

### Product Recommendations

```python
response, _ = agent.generate_response(
    "I'm looking for a laptop for gaming under $1500", 
    session_id="gamer_123"
)
```

### Product Comparisons

```python
response, _ = agent.generate_response(
    "Compare iPhone and Samsung Galaxy phones", 
    session_id="phone_buyer"
)
```

### Price Inquiries

```python
response, _ = agent.generate_response(
    "What's the cheapest smartphone you have?", 
    session_id="budget_shopper"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the example implementations

## Changelog

### v1.0.0
- Initial release
- Basic shopping agent functionality
- Vector-based product search
- Conversation memory
- Session management