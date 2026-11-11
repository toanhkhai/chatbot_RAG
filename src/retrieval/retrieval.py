import json
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MockLLM

# 1. Cấu hình Embedding Model cho truy vấn Vector
Settings.embed_model = HuggingFaceEmbedding(
    model_name="AITeamVN/Vietnamese_Embedding_v2",
    device=0
)

# 2. Nhánh 1: Vector Retriever (Lấy từ ChromaDB)
db = chromadb.PersistentClient(path="vector_database/chroma_db")
chroma_collection = db.get_collection("chatbot_luan_van")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
vector_retriever = index.as_retriever(similarity_top_k=10)

# 3. Nhánh 2: BM25 Retriever (Lấy từ nodes.jsonl) - Các node lưu trong ổ cứng
nodes_path = Path("data/chunks/nodes.jsonl")
nodes = []
with open(nodes_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            nodes.append(TextNode.from_dict(json.loads(line)))

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes,similarity_top_k=10)

# 4. Gộp Hybrid Search bằng RRF
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
    llm=MockLLM()
)

# 5. Khởi tạo Reranker (Cross-Encoder rút từ 10 xuống 4 nodes)
reranker = SentenceTransformerRerank(
    model="AITeamVN/Vietnamese_Reranker",
    top_n=3,
    device="cuda:0"
)
