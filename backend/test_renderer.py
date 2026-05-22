import logging
from dotenv import load_dotenv
load_dotenv()
from services.renderer_processor import SmartClipRenderer

logging.basicConfig(level=logging.INFO)

def main():
    video_id = "MOV_54BC25A5"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    
    print("Starting Smart Clip Rendering Engine Test...")
    processor = SmartClipRenderer(output_base_dir=output_dir)
    results = processor.process_render(video_id, scene_metadata_path)
    print("Success! Rendering Results:", results)

if __name__ == "__main__":
    main()
