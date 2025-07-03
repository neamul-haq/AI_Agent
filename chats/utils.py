# from langchain_openai import ChatOpenAI 
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# import os
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)

# system_prompt = SystemMessage(content="You are a helpful assistant.")

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from .vector import retriever
from django.core.cache import cache

model = OllamaLLM(model="llama3.2")

system_prompt = SystemMessage(
    "You are an expert in answering questions about a pizza restaurant.\n\nHere are some relevant reviews:\n{reviews}\n\nHere is the question to answer:\n{question}"
)


issue_prompt = PromptTemplate.from_template("""
You are an assistant that checks if a user message contains a complaint or issue.

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

    history_msgs.append(HumanMessage(content=user_input))
    prompt = ChatPromptTemplate.from_messages(
        [system_prompt] + history_msgs + [HumanMessage(content="{user_input}")]
    )
    chain = prompt | model

    reviews = retriever.invoke(user_input)

    ai_response = chain.invoke({
        "reviews": reviews,
        "question": user_input,
        "user_input": user_input
    })

    # Detection of issue
    issue_result = issue_chain.invoke({"message": user_input})
    detected_issue = issue_result.strip()

    if "NO_ISSUE" in detected_issue.strip().upper():
        # print(" ✅ NO issue")
        detected_issue = None

    return ai_response, detected_issue