from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from chats.vector import retriever
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()
# model = OllamaLLM(model="llama3.2")
model = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# model = init_chat_model("deepseek-r1-distill-llama-70b")
# print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

issue_prompt = PromptTemplate.from_template("""
"You are an assistant that checks if a customer message contains a complaint or problem related to a shopping site (e.g., product issues, delivery problems, payment errors, etc.)."

Message: "{message}"

If the message includes an issue, summarize it in one sentence.
If it's not an issue, just return "NO_ISSUE".
""")

issue_chain = issue_prompt | model


def generate_response(user_input, history, session_id):
    history_msgs = []

    for msg in history:
        if msg.sender == 'human':
            history_msgs.append(HumanMessage(content=msg.text))
        elif msg.sender == 'ai':
            history_msgs.append(AIMessage(content=msg.text))

    # Add the current user input
    history_msgs.append(HumanMessage(content=user_input))

    # Alternative approach using template variables
    # Create prompt template with placeholders
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are the shop assistant of this shopping site.
Here are our products: {products}

Please help customers with their questions about our products, provide recommendations, 
answer questions about features, pricing, and availability.
""")
    ] + history_msgs)
    
    # Create the chain
    chain = prompt | model
    
    # Generate response
    ai_response = chain.invoke({"products": retriever.invoke(user_input)})
    
    # Detection of issue
    issue_result = issue_chain.invoke({"message": user_input})
    detected_issue = issue_result  # Get the actual string from AIMessage

    if "NO_ISSUE" in detected_issue.content:
        detected_issue = None

    return ai_response.content, detected_issue