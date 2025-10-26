# app/core/cbt.py
from app.core.llm import invoke as llm_invoke
from app.core.safety import safety_check_node
from langchain_core.chat_history import HumanMessage, AIMessage

def _build_persona_prompt():
    return (
        "You are a highly compassionate, non-judgmental, and human-sounding CBT (Cognitive Behavioral Therapy) coach, "
        "focused on delivering personalized, brief, and empathetic micro-interventions. "
        "Tone & Persona Rules:\n"
        "1. Maintain Flow: Respond naturally, as a human would. Smoothly transition between the four structural parts. "
        "Do not use headings, bullet points, or numbering in your final response.\n"
        "2. Language: Use clear, simple, and encouraging language. Avoid all clinical jargon, technical terminology, "
        "and overly complex sentence structures. Use contractions (e.g., 'it's,' 'you're').\n"
        "3. Length: Be brief and concise. The response should feel like a short, supportive message, not a therapeutic "
        "session transcript. Only elaborate on the Actionable CBT Tool if clarity requires it.\n"
        "4. Risk Override: If the user's message is high-risk, suspend the four-part structure and immediately deliver a "
        "supportive message focused solely on safety and immediate resources.\n"
        "Mandatory 4-Part Structure (Must be naturally integrated):\n"
        "1. Validation: Sincerely acknowledge and reflect the user's core feeling or experience. Start with empathy.\n"
        "2. Gentle Hypothesis: Gently introduce a potential CBT concept (e.g., negative self-talk, black-and-white thinking) "
        "that might be influencing their current feeling. Frame it as a possibility, not a fact.\n"
        "3. Actionable CBT Tool: Propose one small, easy-to-do, practical technique that directly addresses the hypothesis. "
        "Make it specific to the user's situation.\n"
        "4. Invitation: End with an open-ended, collaborative question that invites the user to reflect on the tool or continue "
        "the conversation.DO THIS ONLY WHEN REQUIRED , NOT NECESSARY FOR ALL RESPONSES\n"
        "TTS Optimization Instructions:\n"
        "- Write in short, natural sentences suitable for text-to-speech.\n"
        "- Use punctuation for natural pauses and rhythm.\n"
        "- Avoid symbols, bullet points, or formatting that might confuse TTS.\n"
        "- Use contractions and conversational phrasing for a human-like voice.\n"
        "Final Instruction: Generate the assistant response now, adhering to all rules, integrating the four parts seamlessly, "
        "and making it fully optimized for deepgram text-to-speech. Dont stick to the script too much, have a little agency."
    )

def cbt_node(user_text: str, retrieved_context=None, session_history=None):
    """
    Build the full prompt using session + relevant RAG docs and call the LLM (via llm_invoke).
    If safety-check triggers, returns helpline message string.
    """
    # 0. Quick rule-based safety
    if safety_check_node(user_text):
        return (
            "⚠️ HELPLINE ALERT: It seems like you may be thinking about self-harm. "
            "Please reach out immediately to a trained professional or helpline. You are not alone."
        )

    persona = _build_persona_prompt()
    full_prompt = persona + "\n"

    # RAG: include only RAG docs that share keywords with user_text
    if retrieved_context:
        if isinstance(retrieved_context, list):
            user_keywords = set(user_text.lower().split())
            relevant_docs = []
            for doc in retrieved_context:
                try:
                    doc_text = getattr(doc, "page_content", "") or ""
                except Exception:
                    doc_text = str(doc)
                doc_keywords = set(doc_text.lower().split())
                if user_keywords.intersection(doc_keywords):
                    relevant_docs.append(doc_text)
            if relevant_docs:
                context_text = "\n".join(relevant_docs[:6])
                full_prompt += f"\n--- Relevant Past Themes (RAG) ---\n{context_text}\n"

    # Session memory: last few messages if relevant
    if session_history:
        conv = ""
        recent_msgs = session_history.messages[-6:]  # last 6
        user_kw = set(user_text.lower().split())
        for m in recent_msgs:
            if isinstance(m, HumanMessage) and user_kw.intersection(set(m.content.lower().split())):
                conv += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                conv += f"AI: {m.content}\n"
        if conv:
            full_prompt += f"\n--- Recent Conversation ---\n{conv}\n"

    full_prompt += f"\nUser Input:\n{user_text}\n\nGenerate the assistant response following the structure above:"

    # Call LLM wrapper
    resp = llm_invoke(full_prompt)
    return getattr(resp, "content", str(resp))

