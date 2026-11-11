import json
from pathlib import Path
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

input_path = Path("data/chunks/nodes.jsonl")

#Chuyển Node dạng dict sang Node object
nodes = []
with open(input_path,"r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():                
            continue
        data = json.loads(line)             # chuỗi JSON → dict
        node = TextNode.from_dict(data)     # dict → TextNode
        nodes.append(node)                  # thêm vào list

embedding_model = HuggingFaceEmbedding(model_name="AITeamVN/Vietnamese_Embedding_v2",device=0)

#phần code của thư viện chroma, tạo một collection mới và khai báo thuật toán vector search.
db = chromadb.PersistentClient(path="vector_database/chroma_db")
my_collection = db.get_or_create_collection(name="chatbot_luan_van",metadata={"hnsw:space": "cosine"})

#bọc chroma lại bằng thư viện Llamaindex, để có thể lưu các node dưới chroma, nếu k có thì k lưu các node dưới chroma được
vector_store = ChromaVectorStore(chroma_collection=my_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#tiến hành lưu các node dưới chroma.
index = VectorStoreIndex(
    nodes = nodes,
    storage_context=storage_context,
    embed_model=embedding_model,
    show_progress=True
)
