import logging
import os
import tempfile
import soundfile as sf
import json
import torch

from Implicit import extract_implicit_features, voice_to_text, extract_prosody_features
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from datasets import load_dataset, Dataset
import whisper

logger = logging.getLogger(__name__)

# ===============================
# Load models on GPU
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if device == "cuda" else -1
)
whisper_model = whisper.load_model("base").to(device)

# ===============================
# Dataset
# ===============================
ds = load_dataset("AbstractTTS/IEMOCAP") 

def filterd_audio(ds):
    """Filter rows with major_emotion == frustrated or angry."""
    filtered = ds["train"].filter(
        lambda x: x["major_emotion"] in ["frustrated", "angry"]
    )
    results = [
        {
            "audio": row["audio"],
            "major_emotion": row["major_emotion"]
        }
        for row in filtered
    ]
    return results

# ===============================
# Batch processing
# ===============================
def process_results(results, whisper_model, embedding_model, sentiment_pipeline, batch_size=8):
    dataset = Dataset.from_list(results)

    # Step 1. Add text + prosody sequentially
    def add_text_and_prosody(example):
        audio_info = example["audio"]
        audio_path = audio_info.get("path")

        if audio_path is None or not os.path.isfile(audio_path):
            arr, sr = audio_info.get("array"), audio_info.get("sampling_rate")
            if arr is None:
                raise FileNotFoundError("No audio data for this item")
            tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmpf.close()
            sf.write(tmpf.name, arr, sr)
            audio_path = tmpf.name

        try:
            text = voice_to_text(audio_path, whisper_model)
            prosody = extract_prosody_features(audio_path, whisper_model)
            example["text"] = text
            example["prosody"] = prosody
        finally:
            if not audio_info.get("path"):
                os.remove(audio_path)

        return example

    dataset = dataset.map(add_text_and_prosody)

    # Step 2. Add embeddings + sentiment in batches
    def add_embeddings_and_sentiment(batch):
        texts = batch["text"]
        embeddings = embedding_model.encode(texts, convert_to_tensor=True, device=embedding_model.device)
        sentiments = sentiment_pipeline(texts, batch_size=batch_size)

        final_tensors = []
        for t, p, s, emb in zip(texts, batch["prosody"], sentiments, embeddings):
            final_tensor = extract_implicit_features(
                t, embedding_model, sentiment_pipeline, p, only_voice_test=True
            )
            final_tensors.append(final_tensor)

        return {"features": final_tensors}

    dataset = dataset.map(add_embeddings_and_sentiment, batched=True, batch_size=batch_size)

    # Step 3. Keep only required fields
    dataset = dataset.remove_columns(["audio"])  # remove audio
    dataset = dataset.select_columns(["major_emotion", "text", "prosody", "features"])

    return dataset.to_list()


# ===============================
# Run
# ===============================
results = filterd_audio(ds)
print(len(results))
processed_results = process_results(results, whisper_model, embedding_model, sentiment_pipeline, batch_size=8)

# ===============================
# Save JSON
# ===============================
def convert(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

with open("results.json", "w") as f:
    json.dump(processed_results, f, indent=2, default=convert)

print("✅ Saved only emotion, text, prosody, features → results.json")