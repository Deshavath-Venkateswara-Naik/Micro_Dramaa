import os
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OCR Text Analysis"]
)

class OCRProcessRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-ocr")
async def process_ocr(request: OCRProcessRequest):
    """
    Stage 4.6: OCR Text Analysis Engine
    Extracts on-screen text (burned-in subtitles, title cards) from extracted frames.
    """
    try:
        video_path = Path(request.video_path)
        metadata_path = Path(request.scene_metadata_path)
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found at {request.video_path}")
            
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"Metadata file not found at {request.scene_metadata_path}")
            
        output_dir = metadata_path.parent
        
        processor = OCRProcessor(str(output_dir))
        
        results = processor.process_ocr(str(metadata_path))
        
        return {
            "status": "completed",
            "message": f"Stage 4.6 OCR Analysis completed for {request.video_id}",
            "output_dir": str(output_dir),
            "processed_scenes": len(results),
            "ocr_results": results
        }
        
    except Exception as e:
        logger.error(f"Stage 4.6 processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
