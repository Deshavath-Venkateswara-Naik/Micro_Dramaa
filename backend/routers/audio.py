from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
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
        
        logger.info(f"Successfully processed audio energy for {request.video_id}")
        
        return {
            "status": "completed",
            "message": f"Stage 3 audio processing completed for {request.video_id}",
            "output_dir": output_dir,
            "energy_json_path": energy_json_path
        }
    except Exception as e:
        logger.error(f"Error processing audio for {request.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

class SoundEventRequest(BaseModel):
    video_id: str
    video_path: str

@router.post("/extract-sound-events")
async def extract_sound_events(request: SoundEventRequest):
    import subprocess
    from services.script_formatter import CinematicScriptFormatter
    
    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=404, detail="Video path not found")
        
    output_dir = os.path.dirname(request.video_path)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, "full_audio.wav")
    
    try:
        # Extract audio if it doesn't exist
        if not os.path.exists(audio_path):
            logger.info(f"Extracting audio for {request.video_id} to {audio_path}")
            cmd = [
                "ffmpeg", "-y", "-i", request.video_path,
                "-q:a", "0", "-map", "a", audio_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        # Run SED
        logger.info(f"Running Sound Event Detection for {request.video_id}")
        formatter = CinematicScriptFormatter(output_base_dir=output_dir)
        sound_events = formatter.detect_sound_events(audio_path)
        
        # Save to sound_events.json
        import json
        sed_path = os.path.join(output_dir, "sound_events.json")
        data = {
            "sound_events": sound_events
        }
        
        with open(sed_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Saved {len(sound_events)} sound events to {sed_path}")
        
        return {
            "status": "completed",
            "video_id": request.video_id,
            "message": "Sound events extracted and saved to sound_events.json",
            "sound_events": sound_events
        }
    except Exception as e:
        logger.error(f"Error extracting sound events for {request.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SED extraction failed: {str(e)}")
