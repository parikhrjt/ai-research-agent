"""Generate a text-based sample PDF for testing ingestion."""

from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit("Install fpdf2: pip install fpdf2")


def generate(output_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    sections = [
        ("Research Brief: Retrieval-Augmented Generation", "bold"),
        ("", "normal"),
        ("Executive Summary", "bold"),
        (
            "Retrieval-Augmented Generation (RAG) combines information retrieval with "
            "large language model generation to produce factual, grounded answers. "
            "This brief outlines the architecture decisions for our research agent platform.",
            "normal",
        ),
        ("", "normal"),
        ("Key Findings", "bold"),
        (
            "1. Local embedding models (all-MiniLM-L6-v2) achieve 87% answer relevance "
            "on internal document benchmarks without API costs.\n"
            "2. Format-aware chunking improves retrieval precision by 12% over "
            "fixed-size splitting.\n"
            "3. Citation-enforced prompting reduces hallucination rate to under 5% "
            "in manual evaluation.\n"
            "4. pgvector with IVFFlat indexing handles 100K+ chunks with sub-50ms "
            "retrieval latency.",
            "normal",
        ),
        ("", "normal"),
        ("Recommendations", "bold"),
        (
            "- Default to local Ollama inference for development; use OpenAI for production QA.\n"
            "- Set retrieval top-K to 5 with a 0.3 similarity threshold.\n"
            "- Archive all uploaded documents for compliance audit trails.\n"
            "- Evaluate with RAGAS metrics before production deployment.",
            "normal",
        ),
        ("", "normal"),
        ("Contact: research-team@example.com | Version 1.0 | 2024-06-15", "normal"),
    ]

    for text, style in sections:
        if not text:
            pdf.ln(4)
            continue
        if style == "bold":
            pdf.set_font("Helvetica", style="B", size=12 if text.startswith("Research") else 11)
        else:
            pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent.parent / "data" / "samples" / "research_brief.pdf")
