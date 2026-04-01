"""Batch Whisper ASR for CineBrain audio files (.ac3 → text)."""
import os, sys, json, glob, subprocess, tempfile, torch
import numpy as np
from tqdm import tqdm

AUDIO_DIR = "/public/home/maoyaoxin/xxt/datasets/audio_all"
OUTPUT_PATH = "/public/home/maoyaoxin/xxt/datasets/audio_transcripts.json"
MODEL_NAME = "openai/whisper-large-v3"
SAMPLE_RATE = 16000
BATCH_SIZE = 8

def load_audio_ffmpeg(path, sr=16000):
    """Load audio file to numpy array using ffmpeg."""
    ffmpeg_bin = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"
    cmd = [ffmpeg_bin, "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ar", str(sr), "-ac", "1", "-v", "quiet", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return None
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio

def main():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Get all audio files
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.ac3")))
    print(f"Found {len(audio_files)} audio files")

    # Resume from existing
    transcripts = {}
    if os.path.exists(OUTPUT_PATH):
        transcripts = json.load(open(OUTPUT_PATH))
        print(f"Resuming: {len(transcripts)} already done")

    count = 0
    for af in tqdm(audio_files, desc="Whisper ASR"):
        clip_id = os.path.splitext(os.path.basename(af))[0]
        if clip_id in transcripts:
            continue

        audio = load_audio_ffmpeg(af, SAMPLE_RATE)
        if audio is None or len(audio) < 1600:
            transcripts[clip_id] = ""
            continue

        # Truncate to 30s max
        max_len = 30 * SAMPLE_RATE
        audio = audio[:max_len]

        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features, language="en", task="transcribe")
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        transcripts[clip_id] = text
        count += 1

        # Save periodically
        if count % 100 == 0:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(transcripts, f, indent=2)
            print(f"\n  Saved {len(transcripts)} transcripts")

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(transcripts, f, indent=2)
    print(f"\nDone! {len(transcripts)} transcripts saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
