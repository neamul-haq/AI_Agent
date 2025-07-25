from langchain.prompts import ChatPromptTemplate
from .memory import get_summary_memory

class ShoppingAIAgent:
    def __init__(self, model, retriever, issue_checker=None):
        self.model = model
        self.retriever = retriever
        self.issue_checker = issue_checker
        self.memory_store = {}

    def generate_response(self, user_input, session_id):
        memory = get_summary_memory(session_id, self.model, self.memory_store)

        memory.chat_memory.add_user_message(user_input)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are the smart assistant of this shopping site.
Here are our products: {products}

Please help customers with their questions about our products, provide recommendations, 
answer questions shortly about features, pricing, and availability.
""")
        ] + memory.chat_memory.messages)

        chain = prompt | self.model
        result = chain.invoke({"products": self.retriever.invoke(user_input)})
        response_text = getattr(result, "content", "") or getattr(result, "text", "")

        memory.chat_memory.add_ai_message(response_text)

        issue = None
        if self.issue_checker:
            result = self.issue_checker.invoke({"message": user_input})
            content = getattr(result, "content", "")
            if "NO_ISSUE" not in content:
                issue = content

        return response_text, issue


