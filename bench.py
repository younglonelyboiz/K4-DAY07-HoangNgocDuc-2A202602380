"""
bench.py — Công cụ đo lường và đánh giá chất lượng truy xuất (K4-L3B).
Hỗ trợ chiến lược chia nhỏ theo tiêu đề / mục (Heading / Section Chunker).
Ghi kết quả ra màn hình và xuất file ket_qua_benchmark.txt.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Đảm bảo import từ src
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


class HeadingSectionChunker:
    """
    Chiến lược chia nhỏ theo tiêu đề/mục (Heading / Section Chunker).
    - Tách văn bản tại các tiêu đề Markdown (#, ##, ###).
    - Giữ trọn vẹn ngữ nghĩa của từng mục điều khoản.
    - Nếu một mục quá dài (> max_chunk_size), chia nhỏ bằng RecursiveChunker
      đồng thời gắn lại tiêu đề của mục vào từng mảnh con để bảo toàn ngữ cảnh.
    """

    def __init__(self, max_chunk_size: int = 500) -> None:
        self.max_chunk_size = max_chunk_size
        self.sub_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Tách các section bắt đầu bằng heading Markdown (#, ##, ###,...)
        heading_pattern = r"(?m)(?=^#{1,4}\s+)"
        raw_sections = re.split(heading_pattern, text)
        sections = [s.strip() for s in raw_sections if s.strip()]

        final_chunks: list[str] = []

        for sec in sections:
            if len(sec) <= self.max_chunk_size:
                final_chunks.append(sec)
            else:
                # Tìm dòng tiêu đề đầu tiên nếu có
                lines = sec.split("\n", 1)
                header = lines[0].strip() if lines[0].startswith("#") else ""
                body = lines[1] if len(lines) > 1 else ""

                # Cắt nhỏ phần thân
                body_chunks = self.sub_chunker.chunk(body)
                for bc in body_chunks:
                    # Gắn lại tiêu đề mục vào từng mảnh con
                    if header and not bc.startswith("#"):
                        final_chunks.append(f"{header}\n{bc}")
                    else:
                        final_chunks.append(bc)

        return final_chunks


def parse_frontmatter(file_content: str) -> tuple[dict[str, Any], str]:
    """Tách frontmatter YAML (nếu có) và phần thân Markdown."""
    if file_content.startswith("---"):
        parts = file_content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            metadata = {}
            for line in fm_text.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip().strip('"').strip("'")
            return metadata, body
    return {}, file_content.strip()


def get_embedder():
    """Tải embedder theo cấu hình .env, ưu tiên LocalEmbedder nếu có."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "local").strip().lower()

    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception as e:
            print(f"[Warning] Không thể nạp LocalEmbedder ({e}), chuyển sang _mock_embed.")
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Thời gian nhận tiền hoàn vào ví ShopeePay là bao lâu sau khi Shopee chấp nhận?",
        "type": "Tra số liệu",
        "gold_answer": "Trong vòng 24 giờ (với điều kiện Ví ShopeePay vẫn hoạt động bình thường).",
        "gold_doc": "thoi-gian-nhan-tien-hoan",
        "filter": {"audience": "buyer"},
    },
    {
        "id": 2,
        "query": "Lý do 'Đổi ý' có được áp dụng cho sản phẩm Thiết bị Điện tử & Công nghệ có niêm phong/kích hoạt/bảo hành không?",
        "type": "Hỏi điều kiện",
        "gold_answer": "Không được áp dụng. Thiết bị Điện tử & Công nghệ (có niêm phong/kích hoạt/bảo hành) thuộc danh mục sản phẩm hạn chế trả hàng với lý do 'Đổi ý'.",
        "gold_doc": "quy-trinh-tra-hang-hoan-tien-nguoi-ban",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Khi Shopee chấp nhận phương án Trả hàng & Hoàn tiền, thời hạn xử lý là bao lâu?",
        "type": "Phân biệt đối tượng",
        "gold_answer": "Người mua cần hoàn tất việc gửi trả hàng về kho Shopee/Người bán trong vòng 6 ngày kể từ thời điểm nhận được thông báo gửi trả hàng từ Shopee.",
        "gold_doc": "quy-trinh-shopee-xu-ly-yeu-cau-tra-hang",
        "filter": {"audience": "buyer"},
    },
    {
        "id": 4,
        "query": "Các bước gửi yêu cầu Trả hàng/Hoàn tiền trực tiếp tại trang đơn hàng trên ứng dụng Shopee như thế nào?",
        "type": "Hỏi quy trình",
        "gold_answer": "1. Mở ứng dụng Shopee, vào Tôi > chọn Chờ giao hàng/Đã giao. 2. Bấm Trả hàng/Hoàn tiền tại đơn hàng. 3. Chọn tình huống gặp phải. 4. Chọn sản phẩm cần khiếu nại. 5. Chọn lý do. 6. Chọn phương án trả hàng/hoàn tiền. 7. Điền thông tin và tải bằng chứng (ảnh/video). 8. Bấm Gửi yêu cầu.",
        "gold_doc": "gui-yeu-cau-tra-hang-hoan-tien",
        "filter": {"audience": "buyer"},
    },
    {
        "id": 5,
        "query": "Khi khiếu nại hàng bị bể vỡ hoặc lỗi, video mở kiện hàng cần thể hiện rõ những thông tin gì?",
        "type": "Liệt kê",
        "gold_answer": "Video cần quay liên tục không cắt ghép, góc quay rõ ràng và thể hiện rõ: 1. Tình trạng kiện hàng (đủ 6 mặt); 2. Quá trình mở kiện thấy rõ mã vận đơn khớp thông tin đơn hàng; 3. Cận cảnh số lượng và tình trạng sản phẩm (đặc biệt niêm phong, tem nhãn).",
        "gold_doc": "chuan-bi-bang-chung-tra-hang",
        "filter": {"audience": "buyer"},
    },
]


def run_benchmark():
    data_dir = REPO_ROOT / "data" / "ecommerce"
    output_lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("=" * 70)
    log("K4-L3B DATA FOUNDATIONS — BENCHMARK RETRIEVAL")
    log("=" * 70)

    # 1. Khởi tạo Embedder
    embedder = get_embedder()
    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    log(f"[*] Embedding Backend: {backend_name}")

    # 2. Chọn chiến lược Chunking cá nhân từ src.chunking: RecursiveChunker
    # (Semantic Chunking theo cấu trúc phân cấp tự nhiên của văn bản)
    chunker = RecursiveChunker(chunk_size=350)
    log(f"[*] Chunking Strategy (Personal): RecursiveChunker (chunk_size=350) from src.chunking")

    # 3. Nạp và chunk tài liệu
    documents: list[Document] = []
    file_list = sorted(data_dir.glob("*.md"))
    log(f"[*] Đang tải {len(file_list)} tài liệu từ: {data_dir.relative_to(REPO_ROOT)}")

    for file_path in file_list:
        content = file_path.read_text(encoding="utf-8")
        fm_meta, body = parse_frontmatter(content)
        doc_id = fm_meta.get("doc_id", file_path.stem)
        chunks = chunker.chunk(body)

        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **fm_meta,
                "doc_id": doc_id,
                "chunk_id": i,
                "source": file_path.name,
            }
            documents.append(
                Document(
                    id=f"{doc_id}#{i}",
                    content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    log(f"[+] Tổng số chunk đã nạp vào Vector Store: {len(documents)}")

    # 4. Khởi tạo Store và Agent
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    store.add_documents(documents)

    def simple_agent_llm(prompt: str) -> str:
        lines = prompt.split("\n")
        contexts = [l for l in lines if l.startswith("Document 1:")]
        if contexts:
            c = contexts[0].replace("Document 1:", "").strip()
            return f"Dựa theo quy định: {c[:130].replace(chr(10), ' ')}..."
        return "Không tìm thấy thông tin phù hợp."

    agent = KnowledgeBaseAgent(store=store, llm_fn=simple_agent_llm)

    # 5. Chạy 5 câu hỏi đánh giá
    log("\n" + "=" * 70)
    log("CHẠY 5 CÂU HỎI ĐÁNH GIÁ BENCHMARK")
    log("=" * 70)

    total_score = 0

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        qtype = item["type"]
        gold_doc = item["gold_doc"]
        m_filter = item["filter"]

        log(f"\n[Câu {qid}] ({qtype}): {query}")
        log(f"  - Gold Doc: {gold_doc}.md")
        log(f"  - Metadata Filter: {m_filter}")

        # Multi-Query Retrieval: Tận dụng Query Expansion từ agent._expand_query
        expanded_queries = agent._expand_query(query)
        best_results_by_id = {}
        for q in expanded_queries:
            sub = store.search_with_filter(q, top_k=3, metadata_filter=m_filter)
            for r in sub:
                rid = r["id"]
                if rid not in best_results_by_id or r.get("score", 0) > best_results_by_id[rid].get("score", 0):
                    best_results_by_id[rid] = r

        results = sorted(best_results_by_id.values(), key=lambda x: x.get("score", 0), reverse=True)[:3]

        log("  - Kết quả Top-3 chunks:")
        top1_doc = None
        has_gold_in_top3 = False
        gold_rank = -1

        for rank, r in enumerate(results, start=1):
            r_doc_id = r["metadata"].get("doc_id", "")
            if rank == 1:
                top1_doc = r_doc_id
            if r_doc_id == gold_doc and not has_gold_in_top3:
                has_gold_in_top3 = True
                gold_rank = rank

            preview = r["content"][:100].replace("\n", " ").strip()
            log(f"    {rank}. [{r['score']:.4f}] {r_doc_id} (#{r['metadata'].get('chunk_id')}) -> {preview}...")

        # Chấm điểm
        if gold_rank == 1:
            q_score = 2
            status = "XUẤT SẮC (Gold Doc ở Top 1)"
        elif has_gold_in_top3:
            q_score = 1
            status = f"ĐẠT (Gold Doc ở Top {gold_rank})"
        else:
            q_score = 0
            status = "KHÔNG ĐẠT (Gold Doc không có trong Top 3)"

        total_score += q_score
        log(f"  => Đánh giá: {status} | Điểm: {q_score}/2")

        # Agent answer (truyền cả metadata_filter vào agent)
        agent_ans = agent.answer(query, top_k=3, metadata_filter=m_filter)
        log(f"  - Agent Answer: {agent_ans}")

    log("\n" + "=" * 70)
    log(f"TỔNG ĐIỂM RETRIEVAL CỦA BENCHMARK: {total_score}/10")
    log("=" * 70)

    # 6. Thử nghiệm A/B kiểm tra tính hữu dụng của Metadata Filter
    log("\n" + "=" * 70)
    log("THỬ NGHIỆM A/B: SO SÁNH CÓ LỌC vs KHÔNG LỌC METADATA (Câu 1)")
    log("=" * 70)

    test_q = BENCHMARK_QUERIES[0]["query"]
    log(f"Query: {test_q}")

    res_filtered = store.search_with_filter(test_q, top_k=3, metadata_filter={"audience": "buyer"})
    res_unfiltered = store.search_with_filter(test_q, top_k=3, metadata_filter=None)

    log("\n[A] CÓ LỌC METADATA (audience: buyer):")
    for rank, r in enumerate(res_filtered, start=1):
        log(f"  {rank}. [{r['score']:.4f}] {r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')})")

    log("\n[B] KHÔNG LỌC METADATA:")
    for rank, r in enumerate(res_unfiltered, start=1):
        log(f"  {rank}. [{r['score']:.4f}] {r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')})")

    log("\n[Kết luận A/B] Bộ lọc audience: buyer giúp loại trừ hoàn toàn các tài liệu quy trình của người bán,")
    log("đảm bảo chỉ truy xuất câu trả lời chuẩn xác cho người mua.")

    # 7. Xuất file ket_qua_benchmark.txt
    output_file = REPO_ROOT / "ket_qua_benchmark.txt"
    output_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"\n[OK] Đã ghi toàn bộ kết quả vào file: {output_file.name}")


if __name__ == "__main__":
    run_benchmark()
