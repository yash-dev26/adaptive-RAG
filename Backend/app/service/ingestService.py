from app.ingestion.embeddings import gen_embeddingsAndStoreInQdrant
from app.ingestion.chunking import load_file, split_text
from app.config.server import config

async def ingest_data(request):
    # if request.pdf_file:
    #     text = await extract_text_from_pdf(request.pdf_file)
    # else:
    #     text = request.text

    text = load_file(request.file_path)
    print(f"Loaded text from file: {text[:100]}...")  # Debug: Print the first 100 characters of the loaded text

    chunks = split_text(text)
    

    result = await gen_embeddingsAndStoreInQdrant(chunks, request.file_id, request.user_id)
    return result
