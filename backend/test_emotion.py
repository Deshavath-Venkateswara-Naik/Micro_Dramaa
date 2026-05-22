from services.emotion_processor import EmotionIntelligenceProcessor
import logging

logging.basicConfig(level=logging.INFO)

def main():
    video_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/standardized_video.mp4"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    
    print("Starting Emotion Intelligence Test...")
    processor = EmotionIntelligenceProcessor(output_base_dir=output_dir)
    processor.process_emotion(video_path, scene_metadata_path)
    print("Success!")

if __name__ == "__main__":
    main()
