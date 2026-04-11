from typing import List
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text

def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(text)
    return chunks

    