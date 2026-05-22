import logging
from dotenv import load_dotenv
load_dotenv()
from services.continuous_processor import ContinuousIntelligenceProcessor
from services.renderer_processor import SmartClipRenderer

logging.basicConfig(level=logging.INFO)

def main():
    video_id = "MOV_54BC25A5"
    scene_metadata_path = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/scene_metadata.json"
    output_dir = "/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5"
    
    print("Starting Continuous Micro-Drama Generation (Stage 14)...")
    processor = ContinuousIntelligenceProcessor(output_base_dir=output_dir)
    results = processor.process_continuous_story(video_id, scene_metadata_path)
    print("Continuous Episodic Array Generated:", len(results), "episodes.")
    
    print("Starting Smart Batch Renderer...")
    renderer = SmartClipRenderer(output_base_dir=output_dir)
    render_results = renderer.process_render(video_id, scene_metadata_path)
    print("Render Results:", render_results)

if __name__ == "__main__":
    main()
