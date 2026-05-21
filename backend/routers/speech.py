from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from services.speech_processor import SpeechProcessor
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class SpeechProcessRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-speech")
async def process_speech(request: SpeechProcessRequest):
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video path not found")
        
    if not os.path.exists(request.scene_metadata_path):
        raise HTTPException(status_code=404, detail="Scene metadata not found")

    output_dir = os.path.dirname(request.video_path)
    
    try:
        processor = SpeechProcessor(output_base_dir=output_dir)
        results = processor.process_speech(request.video_path, request.scene_metadata_path)
        logger.info(f"Successfully processed speech for {request.video_id}")
        
        return {
            "status": "completed",
            "message": f"Stage 4 Speech Intelligence completed for {request.video_id}",
            "output_dir": output_dir,
            "processed_scenes": len(results),
            "intelligence_results": results
        }
    except Exception as e:
        logger.error(f"Error processing speech for {request.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech processing failed: {str(e)}")
