import logging
from dotenv import load_dotenv
load_dotenv()
from services.drama_scorer_processor import DramaScoringProcessor

logging.basicConfig(level=logging.INFO)

def main():
    video_id = "MOV_54BC25A5"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    genre = "Action"
    
    print(f"Starting Multi-Layer Drama Scoring Test for Genre: {genre}...")
    processor = DramaScoringProcessor(output_base_dir=output_dir)
    results = processor.process_drama_scoring(video_id, scene_metadata_path, genre)
    print("Success! Action Genre Results:", results)
    
    print(f"\nStarting Multi-Layer Drama Scoring Test for Genre: Serials...")
    results_serial = processor.process_drama_scoring(video_id, scene_metadata_path, "Serials")
    print("Success! Serials Genre Results:", results_serial)

if __name__ == "__main__":
    main()
