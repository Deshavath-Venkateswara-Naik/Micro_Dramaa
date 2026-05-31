import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add backend directory to sys.path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from services.gemini_llm_processor import GeminiLLMProcessor

def test():
    load_dotenv(os.path.join(os.path.dirname(__file__), "backend", ".env"))
    
    video_id = "MOV_993D77C0"
    storage_dir = os.path.join(os.path.dirname(__file__), "storage", video_id)
    
    processor = GeminiLLMProcessor(output_base_dir=storage_dir)
    print("Running Gemini LLM Processor...")
    result = processor.process_gemini_llm(video_id=video_id)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    out_path = Path(storage_dir) / "scenes_and_plot.json"
    if out_path.exists():
        print(f"\nSuccessfully created {out_path}")
    else:
        print(f"\nFailed to create {out_path}")

if __name__ == "__main__":
    test()
