import pymupdf4llm
import docx2txt
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_path: str):
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return pymupdf4llm.to_markdown(file_path, page_chunks=True)

    if suffix == ".txt":
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return [{"text": text, "metadata": {"page": 0}}]

    if suffix == ".docx":
        text = docx2txt.process(file_path)
        return [{"text": text, "metadata": {"page": 0}}]

    raise ValueError(f"Unsupported file type: {suffix}")

def split_text(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for idx, page in enumerate(pages):
        # Handle pages with or without metadata
        if isinstance(page, dict):
            metadata = page.get("metadata", {})
            page_num = metadata.get("page", idx) + 1
            text = page.get("text", page) if "text" in page else str(page)
        else:
            # If page is not a dict, treat it as text
            page_num = idx + 1
            text = str(page)

        split_chunks = splitter.split_text(text)

        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "page": page_num,
            })

    return chunks