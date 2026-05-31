import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
from services.gemini_llm_processor import GeminiLLMProcessor

router = APIRouter()

class GeminiLLMRequest(BaseModel):
    video_id: str

@router.post("/gemini-llm", tags=["Gemini LLM"])
async def process_gemini_llm(request: GeminiLLMRequest):
    """
    Analyzes multiple modalities (shots, volume energy, diarization, clip embeddings)
    using Gemini LLM to generate narrative scenes and the movie plot.
    """
    try:
        storage_path = os.path.join(os.path.dirname(__file__), "..", "..", "storage", request.video_id)
        if not os.path.exists(storage_path):
            raise HTTPException(status_code=404, detail=f"Storage directory not found for video {request.video_id}")
            
        processor = GeminiLLMProcessor(output_base_dir=storage_path)
        result = processor.process_gemini_llm(video_id=request.video_id)
        
        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("scenes_and_plot", {}).get("error", "Unknown error"))
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
