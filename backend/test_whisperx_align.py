import whisperx
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, meta = whisperx.load_align_model(language_code="te", device=device)
audio = whisperx.load_audio("/home/venkateswara/Micro_Drama/storage/MOV_54BC25A5/audio/dialogue/dialogue_SC_001.wav")
duration = audio.shape[0] / 16000
text = "Nenu okkasari commit ayithe na maata nene vinanu."

try:
    result = whisperx.align(
        [{"text": text, "start": 0.0, "end": duration}],
        model,
        meta,
        audio,
        device,
        return_char_alignments=False
    )
    print("Alignment success!")
    print(result["word_segments"])
except Exception as e:
    print("Error:", repr(e))
