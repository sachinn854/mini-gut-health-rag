from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_rag_prompt():
    """
    Create RAG prompt template with conversation history support.
    
    Returns:
        ChatPromptTemplate with system message, chat history, and user query
    """
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful medical assistant specializing in gut health and microbiome.

CRITICAL GROUNDING RULES:
1. You MUST answer ONLY using information EXPLICITLY stated in the provided context below
2. If the context does NOT mention the topic, respond EXACTLY with: "I don't know based on the provided context."
3. Do NOT use your prior knowledge or training data
4. Do NOT infer or extrapolate beyond what is written in the context
5. Do NOT make assumptions or connections not explicitly stated

ANSWER QUALITY RULES:
6. Answer COMPREHENSIVELY using ALL relevant information in the context
7. If multiple conditions, factors, or examples are mentioned, LIST ALL of them explicitly
8. Provide a COMPLETE answer rather than a partial one
9. Extract and include all relevant details, not just the first example
10. Answer in COMPLETE, grammatically correct sentences

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    return template

def build_prompt(query, docs, chat_history=[]):
    """
    Build a simple prompt from query and retrieved documents.
    For non-chain usage.
    
    Args:
        query: User question
        docs: Retrieved documents
        chat_history: Previous conversation messages (optional)
        
    Returns:
        Formatted prompt string
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format chat history if present
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in chat_history[-4:]:  # Last 2 exchanges
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"
    
    prompt = f"""You are a medical assistant specializing in gut health.

CRITICAL GROUNDING RULES:
- You MUST answer ONLY using information EXPLICITLY stated in the provided context below
- If the context does NOT mention the topic, respond EXACTLY with: "I don't know based on the provided context."
- Do NOT use your prior knowledge or training data
- Do NOT infer or extrapolate beyond what is written in the context
- Do NOT make assumptions or connections not explicitly stated in the context

ANSWER QUALITY RULES:
- Answer COMPREHENSIVELY using ALL relevant information in the context
- If multiple conditions, factors, or examples are mentioned, LIST ALL of them explicitly
- Provide a COMPLETE answer rather than a partial one
- Extract and include all relevant details, not just the first example
- Answer in COMPLETE, grammatically correct sentences

Context:
{context}
{history_text}

Question: {query}

Answer (comprehensive and complete, listing ALL relevant information from context):"""
    
    return prompt
