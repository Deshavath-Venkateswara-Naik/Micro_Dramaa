from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from services.audio_processor import AudioProcessor
from services.audio_energy import analyze_audio_energy
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class AudioProcessRequest(BaseModel):
    video_id: str
    video_path: str
    scene_metadata_path: str

@router.post("/process-audio")
async def process_audio(request: AudioProcessRequest):
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video path not found")
        
    if not os.path.exists(request.scene_metadata_path):
        raise HTTPException(status_code=404, detail="Scene metadata not found")

    output_dir = os.path.dirname(request.video_path)
    
    try:
        # Extract audio, calculate energy per second, and save to storage folder
        energy_json_path = analyze_audio_energy(request.video_id, request.video_path, output_dir)
        
        # Existing AudioProcessor logic
        processor = AudioProcessor(output_base_dir=output_dir)
        results = processor.process_movie(request.video_path, request.scene_metadata_path)
        logger.info(f"Successfully processed audio for {request.video_id}")
        
        return {
            "status": "completed",
            "message": f"Stage 3 audio processing completed for {request.video_id}",
            "output_dir": output_dir,
            "processed_scenes": len(results),
            "energy_json_path": energy_json_path
        }
    except Exception as e:
        logger.error(f"Error processing audio for {request.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
