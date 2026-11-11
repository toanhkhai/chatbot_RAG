import os
import json
from dotenv import load_dotenv
from pathlib import Path
from llama_cloud import LlamaCloud
from llama_index.core.schema import Document

load_dotenv()
api_key = os.environ["LLAMA_PARSE_API_KEY"]

document_raw_path = Path("data/raw/quy_che_hoc_vu.pdf")
document_parsed_path = Path("data/parsed/ketqua.jsonl")

client = LlamaCloud(api_key=api_key)

#upload file lên cloud trước, sau đó mới parse.
upload_file = client.files.create(file=Path(document_raw_path), purpose="parse")
result = client.parsing.parse(
    file_id = upload_file.id,
    tier = "cost_effective",
    version="latest",
    expand=["markdown"]
)

documents: list[Document] = []
for page in result.markdown.pages:
    # Bỏ qua trang parse thất bại
    if page.success is False:
        continue
    # Bỏ qua trang không có nội dung markdown
    if page.markdown is None:
        continue
    if page.markdown.strip() == "":
        continue
    doc = Document(
        text=page.markdown,
        metadata={
            "page_number": page.page_number,
            "source_file": document_raw_path.name,
        },
    )
    documents.append(doc)

# Lưu danh sách Document
document_parsed_path.parent.mkdir(parents=True, exist_ok=True)
with open (document_parsed_path,"w",encoding="utf-8") as f:
    for doc in documents:
        line_json = json.dumps(doc.to_dict(), ensure_ascii=False)
        f.write(line_json + "\n")

    