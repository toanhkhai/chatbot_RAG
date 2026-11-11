import json
from pathlib import Path
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

jsonl_path = Path("data/parsed/ketqua.jsonl")
documents = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        # 1. Load chuẩn đối tượng Document của LlamaIndex
        doc = Document.from_dict(json.loads(line))
        # Không cho tên file làm nhiễu vector khi nhúng
        doc.excluded_embed_metadata_keys = ["source_file"]
        documents.append(doc)

print(f"Đã đọc {len(documents)} trang tài liệu từ file JSONL.")

# 2. Băm thông minh cho văn bản Quy chế (Markdown):
# Bước 1: Tách theo mục (#, ##, ###) để giữ trọn vẹn từng Điều/Khoản và Bảng
# Bước 2: Đoạn nào vẫn dài hơn 512 token thì mới cắt tiếp bằng SentenceSplitter
pipeline = IngestionPipeline(
    transformations=[
        MarkdownNodeParser(),
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
    ]
)
nodes: list[BaseNode] = pipeline.run(documents=documents)

# 3. Gán ID cố định để không bao giờ bị trùng lặp khi nạp vào ChromaDB
for idx, node in enumerate(nodes):
    node.id_ = f"quy_che_node_{idx}"

output_chunk_path = Path("data/chunks/nodes.jsonl")
output_chunk_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_chunk_path, "w", encoding="utf-8") as f:
    for node in nodes:
        f.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

print(f"Đã tạo thành công {len(nodes)} chunks tại {output_chunk_path}")