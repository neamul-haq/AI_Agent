from langchain.memory import ConversationSummaryMemory

def get_summary_memory(session_id, model, memory_store):
    if session_id not in memory_store:
        memory = ConversationSummaryMemory(llm=model)
        memory_store[session_id] = memory
    return memory_store[session_id]
