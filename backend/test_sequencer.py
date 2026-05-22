import logging
from dotenv import load_dotenv
load_dotenv()
from services.sequencer_processor import DramaSequencerProcessor

logging.basicConfig(level=logging.INFO)

def main():
    video_id = "MOV_54BC25A5"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    
    print("Starting AI Drama Sequencer Test...")
    processor = DramaSequencerProcessor(output_base_dir=output_dir)
    results = processor.process_sequencer(video_id, scene_metadata_path)
    print("Success! Sequencer Results:", results)

if __name__ == "__main__":
    main()
