import logging
from dotenv import load_dotenv
load_dotenv()
from services.nostalgia_processor import NostalgiaIntelligenceProcessor

logging.basicConfig(level=logging.INFO)

def main():
    video_id = "MOV_54BC25A5"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    
    print("Starting Nostalgia Intelligence Test...")
    processor = NostalgiaIntelligenceProcessor(output_base_dir=output_dir)
    results = processor.process_nostalgia(video_id, scene_metadata_path)
    print("Success! Results:", results)

if __name__ == "__main__":
    main()
