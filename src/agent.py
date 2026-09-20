from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def _expand_query(self, question: str) -> list[str]:
        """
        Query Expansion (Mở rộng truy vấn):
        Tạo các biến thể đồng nghĩa của câu hỏi để tăng Recall (độ phủ),
        giúp hệ thống bắt được các tài liệu dùng từ vựng tương đương.
        """
        queries = [question]
        synonyms = {
            "các bước": ["hướng dẫn", "cách"],
            "như thế nào": ["hướng dẫn", "thao tác"],
            "chấp nhận": ["đồng ý", "duyệt"],
            "thời hạn": ["thời gian", "số ngày"],
            "trả hàng": ["đổi trả", "hoàn trả"],
            "hoàn tiền": ["nhận lại tiền", "trả lại tiền"],
            "shopeepay": ["ví shopeepay"],
            "bể vỡ": ["hư hỏng", "hư hại"],
            "người bán": ["shop", "nhà bán"],
            "đóng gói": ["quy cách đóng gói", "bọc hàng"],
        }

        q_lower = question.lower()
        for term, syns in synonyms.items():
            if term in q_lower:
                for syn in syns:
                    variant = q_lower.replace(term, syn)
                    if variant not in queries:
                        queries.append(variant)

        return queries

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        # 1. Query Expansion & Multi-Query Retrieval
        queries = self._expand_query(question)

        # 2. Hợp nhất kết quả từ tất cả các truy vấn mở rộng (Deduplication + Score Fusion)
        best_results_by_id = {}
        for q in queries:
            if metadata_filter:
                sub_results = self.store.search_with_filter(q, top_k=top_k, metadata_filter=metadata_filter)
            else:
                sub_results = self.store.search(q, top_k=top_k)
            for r in sub_results:
                rid = r["id"]
                if rid not in best_results_by_id or r.get("score", 0) > best_results_by_id[rid].get("score", 0):
                    best_results_by_id[rid] = r

        # Sắp xếp lại theo độ liên quan giảm dần và lấy top_k
        merged_results = sorted(best_results_by_id.values(), key=lambda x: x.get("score", 0), reverse=True)
        results = merged_results[:top_k]

        # 3. Contextual Compression: Chỉ giữ lại các chunk có relevance score > 0
        context_parts = []
        for i, r in enumerate(results):
            if r.get("score", 0) > 0 or not r.get("score"):
                context_parts.append(f"Document {i+1}:\n{r['content']}")

        context = "\n\n".join(context_parts)

        # 4. Xây dựng prompt có ngữ cảnh
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # 5. Gọi LLM sinh câu trả lời
        return self.llm_fn(prompt)
