import uuid
from generation.llm import generate_answer

def main():
    # Tự sinh 1 mã phiên ngẫu nhiên khi mở app (thoát app là mất mã này -> mất lịch sử)
    session_id = str(uuid.uuid4())
    print(f"=== CHATBOT QUY CHẾ HỌC VỤ (Session: {session_id[:8]}) ===")
    print("Gõ 'q' để thoát.\n")

    while True:
        question = input("Sinh viên: ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue

        # Gọi hàm xử lý từ llm_client.py
        response = generate_answer(session_id=session_id, question=question)

        # In câu trả lời của AI
        print(f"\nBot: {response.response}\n")

        # In nguồn trích dẫn (trang mấy, điểm rerank bao nhiêu) để bạn kiểm tra độ chuẩn của RAG
        print("--- Nguồn tham khảo ---")
        for i, node in enumerate(response.source_nodes, 1):
            page = node.metadata.get("page_number", "?")
            file_name = node.metadata.get("source_file", "?")
            score = f"{node.score:.4f}" if node.score is not None else "N/A"
            print(f"[{i}] Trang {page} ({file_name}) | Điểm Rerank: {score}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()