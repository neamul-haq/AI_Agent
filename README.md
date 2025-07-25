# shopping_ai_agent

**shopping_ai_agent** is a modular AI Shopping Assistant built with LangChain.  
It allows you to create intelligent product advisors by combining powerful LLMs (like Groq, OpenRouter, or Ollama) with real-time product data and conversational memory.

---

## ✨ Features

- 🧠 Conversational AI shopping assistant
- 🔌 Plug-and-play support for OpenAI-compatible LLMs (Groq, Ollama, OpenRouter)
- 🔍 Dynamic product retrieval and embedding from public APIs
- 🗂️ ChromaDB-powered document retriever
- 🧾 ConversationSummaryMemory to retain context
- 🔧 Framework-agnostic: use in any Python environment

---

## 📦 Installation

Install from PyPI (once published):

```bash
pip install shopping-ai-agent
```

Or install from local `.whl`:

```bash
pip install /path/to/shopping_ai_agent-0.1.0-py3-none-any.whl
```

---

## 🚀 Quickstart

```python
from shopping_ai_agent import ShoppingAIAgent, build_chroma_from_api
from langchain_community.chat_models import ChatOpenAI

# Initialize your LLM (Groq in this example)
model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    openai_api_key="your-groq-key",
    model="llama3-70b-8192"
)

# Create retriever from any public API (JSON format)
retriever = build_chroma_from_api(
    api_url="https://fakestoreapi.in/api/products",
    persist_path="./chroma_store"
)

# Create agent with memory
agent = ShoppingAIAgent(model=model, retriever=retriever)

# Ask a question
response, issue = agent.generate_response("What is the cheapest product?", [], session_id="test-1")
print("Response:", response)
print("Detected Issue:", issue)
```

---

## 💡 How It Works

- `ShoppingAIAgent`: Handles user queries, retrieves relevant documents from ChromaDB, and generates a conversational response using the model.
- `build_chroma_from_api`: Fetches and indexes product data from a public API using LangChain documents and embeddings.
- Memory is handled via `ConversationSummaryMemory` to keep interactions context-aware across sessions.

---

## 🧱 Project Structure

```
shopping_ai_agent/
├── __init__.py
├── agent.py
├── memory.py
├── retriever.py
```

---

## 🛠 Requirements

- Python 3.8+
- Dependencies:
  - `langchain`
  - `langchain-community`
  - `langchain-openai`
  - `langchain-chroma`
  - `requests`
  - `tqdm`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Local Testing

Example script to test:

```bash
python test_agent.py
```

Make sure your virtual environment is activated and `.env` contains necessary API keys like:

```env
GROQ_API_KEY=your_key_here
```

---

## 👤 Author

**Neamul Haq**  
GitHub: [@neamulhaq](https://github.com/neamulhaq)
Mail: neamul.cse6.bu@gmail.com
---

## 📄 License

This project is licensed under the MIT License.