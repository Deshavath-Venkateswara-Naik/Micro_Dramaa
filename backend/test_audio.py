import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

from services.speech_processor import SpeechProcessor

def main():
    video_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/standardized_video.mp4"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    
    print("Starting manual test...")
    try:
        processor = SpeechProcessor(output_base_dir=output_dir)
        processor.process_speech(video_path, scene_metadata_path)
        print("Success!")
    except Exception as e:
        print(f"FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
