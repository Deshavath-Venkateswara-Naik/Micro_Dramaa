import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

from services.audio_processor import AudioProcessor

def main():
    video_path = "/home/venkateswara/Micro_Drama/storage/MOV_0D744FB2/standardized_video.mp4"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_0D744FB2/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_0D744FB2"
    
    print("Starting manual test...")
    try:
        processor = AudioProcessor(output_base_dir=output_dir)
        processor.process_movie(video_path, scene_metadata_path)
        print("Success!")
    except Exception as e:
        print(f"FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
