import time
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import os
from dotenv import load_dotenv

load_dotenv("backend/.env")

audio_file = "/home/venkateswara/Micro_Drama/storage/MOV_993D77C0/audio/chunks/chunk_004.wav"

t0 = time.time()
audio = whisperx.load_audio(audio_file)
hf_token = os.environ.get("HF_TOKEN")
model = DiarizationPipeline(token=hf_token, device="cpu")
res = model(audio)
print(f"Diarization took: {time.time() - t0} seconds")
print(res)
