from fastapi import APIRouter, UploadFile, File, HTTPException
from src.models.schemas import DocumentBase
import os

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf') and not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and TXT are supported.")
    
    # Placeholder for ingestion logic
    # 1. Save file locally
    # 2. Extract text
    # 3. Create chunks
    # 4. Store in Vector DB
    
    return {"filename": file.filename, "status": "processed"}
