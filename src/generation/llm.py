import os
from threading import Lock
from dotenv import load_dotenv
from cachetools import TTLCache

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.google_genai import GoogleGenAI

from retrieval.retrieval import hybrid_retriever, reranker
from generation.prompt import SYSTEM_PROMPT, CONTEXT_PROMPT, condense_chat_Prompt

load_dotenv()

# 1. Khởi tạo Gemini LLM
Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.1
)

# 2. Quản lý Session Memory trên RAM
session_memories = TTLCache(maxsize=500, ttl=1800)
cache_lock = Lock()

def get_session_memory(session_id: str) -> ChatMemoryBuffer:
    with cache_lock:
        memory = session_memories.get(session_id)
        if memory is None:
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            # (Tùy chọn) Mồi sẵn 1 câu chào của ASSISTANT vào lịch sử khi mới mở phiên:
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Chào bạn! Mình là trợ lý tư vấn Quy chế Học vụ. Bạn cần tra cứu quy định nào?"
                )
            )
        session_memories[session_id] = memory
        return memory

# 3. Khởi tạo Chat Engine có phân quyền SYSTEM / USER / ASSISTANT đầy đủ
def generate_answer(session_id: str, question: str):
    memory = get_session_memory(session_id)
    
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        memory=memory,
        system_prompt=SYSTEM_PROMPT,           # Gắn vào MessageRole.SYSTEM
        context_prompt=CONTEXT_PROMPT,         # Nối tiếp vào MessageRole.SYSTEM kèm {context_str}
        condense_prompt=condense_chat_Prompt,  # Dùng ChatPromptTemplate có chia SYSTEM / USER
        verbose=False
    )
    
    # Khi gọi .chat(question):
    # - question tự động được bọc thành ChatMessage(role=MessageRole.USER, content=question)
    # - Câu trả lời của Gemini tự động bọc thành ChatMessage(role=MessageRole.ASSISTANT, ...) và lưu vào memory
    return chat_engine.chat(question)