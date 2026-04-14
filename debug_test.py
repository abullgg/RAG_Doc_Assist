import asyncio
from src.api.routes.upload import upload_document
from src.api.routes.ask import ask_question
from src.models.schemas import AskRequest
from fastapi import UploadFile
import io
import src.main as state

async def main():
    f = UploadFile(filename='requirements.txt', file=io.BytesIO(b'Some random text to test RAG...'))
    res1 = await upload_document(f)
    print("Upload result:", res1)
    print("Chunk store size:", len(state.retrieval_service._chunk_store))
    print("Chunk store keys:", state.retrieval_service._chunk_store.keys())

    req = AskRequest(question='what is the text?', top_k=1)
    res2 = await ask_question(req)
    print("Ask result:", res2)

asyncio.run(main())
