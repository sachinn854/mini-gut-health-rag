from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Define prompt template
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "history", "query"],
    template="""You are a medical assistant specializing in gut health.

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

===== RETRIEVED CONTEXT (Use this to answer) =====
{context}

===== CONVERSATION HISTORY (For reference only) =====
{history}

===== CURRENT QUESTION =====
{query}

Answer (comprehensive and complete, listing ALL relevant information from context):"""
)

def build_prompt(query, docs, chat_history=[]):
    """
    Build a prompt from query and retrieved documents using PromptTemplate.
    
    Args:
        query: User question
        docs: Retrieved documents
        chat_history: Previous conversation messages (optional)
        
    Returns:
        Formatted prompt string
    """
    # Format documents into context
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
    
    # Use PromptTemplate to format
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        history=history_text,
        query=query
    )
    
    return prompt
