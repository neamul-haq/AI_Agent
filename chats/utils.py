from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from chats.vector import retriever
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from chats.models import ChatSession 
# Load environment variables
load_dotenv()
# model = OllamaLLM(model="llama3.2")
model = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# model = init_chat_model("deepseek-r1-distill-llama-70b")
# print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

# Global dictionary for session memories
session_memory_store = {}

def get_summary_memory(session_id):
    if session_id not in session_memory_store:
        session = ChatSession.objects.get(id=session_id)

        memory = ConversationSummaryMemory(
            llm=model,
            # return_messages=True
        )

        # Restore saved summary into memory
        if session.summary:
            memory.buffer = session.summary

        session_memory_store[session_id] = memory

    return session_memory_store[session_id]

issue_prompt = PromptTemplate.from_template("""
"You are an assistant that checks if a customer message contains a complaint or problem related to a shopping site (e.g., product issues, delivery problems, payment errors, etc.)."

Message: "{message}"

If the message includes an issue, summarize it in one sentence.
If it's not an issue, just return "NO_ISSUE".
""")

issue_chain = issue_prompt | model

def generate_response(user_input, history, session_id):
    # Get or create memory for this session
    summary_memory = get_summary_memory(session_id)

    for msg in history:
        if msg.sender == 'human':
            summary_memory.chat_memory.add_user_message(msg.text)
        elif msg.sender == 'ai':
            summary_memory.chat_memory.add_ai_message(msg.text)
    # Add the current user input
    summary_memory.chat_memory.add_user_message(user_input)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are the smart assistant of this shopping site.
Here are our products: {products}

Please help customers with their questions about our products, provide recommendations, 
answer questions shortly about features, pricing, and availability.
""")
    ] + summary_memory.chat_memory.messages)

    chain = prompt | model

    ai_response = chain.invoke({"products": retriever.invoke(user_input)})

    # Add AI response to memory
    summary_memory.chat_memory.add_ai_message(ai_response.content)

    # Save updated summary to DB
    session = ChatSession.objects.get(id=session_id)
    session.summary = summary_memory.buffer
    session.save()


    issue_result = issue_chain.invoke({"message": user_input})
    detected_issue = issue_result  # Get the actual string from AIMessage

    if "NO_ISSUE" in detected_issue.content:
        detected_issue = None

    return ai_response.content, detected_issue