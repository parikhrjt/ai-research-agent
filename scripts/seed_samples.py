"""Seed the vector store with sample documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import setup_logging
from app.rag import get_rag_pipeline
from app.vectorstore import initialize_vector_store

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_FILES = [
    "transformer_architecture.md",
    "rag_pipeline_notes.txt",
    "experiment_results.csv",
    "research_brief.pdf",
]


def main() -> None:
    setup_logging()
    initialize_vector_store()
    pipeline = get_rag_pipeline()

    for fname in SAMPLE_FILES:
        fpath = SAMPLES_DIR / fname
        if not fpath.exists():
            print(f"  SKIP  {fname} (not found — run generate_sample_pdf.py first)")
            continue
        result = pipeline.ingest_path(str(fpath), filename=fname)
        print(f"  OK    {fname} → {result.chunk_count} chunks (id: {result.document_id[:8]}...)")

    print("\nSeeding complete.")


if __name__ == "__main__":
    main()
