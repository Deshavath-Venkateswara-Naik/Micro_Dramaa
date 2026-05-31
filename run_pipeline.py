import os
import sys
import logging

# Add backend to path so we can import services
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.speech_processor import SpeechProcessor

logging.basicConfig(level=logging.INFO)

def run():
    output_base_dir = "/home/venkateswara/Micro_Drama/storage/MOV_993D77C0"
    sp = SpeechProcessor(output_base_dir)
    
    print("Starting speech processing...")
    # This will use the existing global_dialogue.wav and the existing MOV_993D77C0_global_final_merged.json,
    # perform WhisperX alignment, Pyannote diarization, and save to dialogue_diarization.json
    result = sp.process_speech(video_id="MOV_993D77C0", video_path=None)
    
    if result and "dialogues" in result:
        print(f"Successfully processed {len(result['dialogues'])} dialogue segments.")
    else:
        print("Speech processing failed or returned empty results.")

if __name__ == "__main__":
    run()
