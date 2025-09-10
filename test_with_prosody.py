import os
import json
import soundfile as sf
from tuning import get_tone_adapter, get_speaker_embedding, generate_and_store_speech
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from Implicit import extract_prosody_features, voice_to_text, extract_implicit_features
import torch
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if device == "cuda" else -1
)
whisper_model = whisper.load_model("base").to(device)



processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
base_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

def calculate_and_store_mean_of_results(results, save_path):
    """
    Calculate the mean of numerical results and store them in a JSON file.
    Also store the list of all values for each key.
    Skips non-numerical columns except for 'prosody' (which is processed recursively).
    For 'features', calculates mean for each index and stores all lists as well.
    """
    if not results:
        print("No results to process.")
        return

    mean_results = {}
    value_lists = {}

    # Collect all keys except 'text', 'major_emotion', 'gender'
    keys = set()
    for result in results:
        keys.update(result.keys())
    keys.discard('text')
    keys.discard('major_emotion')
    keys.discard('gender')

    # Prepare lists for each key
    for key in keys:
        if key == "prosody":
            # Handle nested prosody keys
            prosody_keys = set()
            for result in results:
                if "prosody" in result and isinstance(result["prosody"], dict):
                    prosody_keys.update(result["prosody"].keys())
            for pkey in prosody_keys:
                value_lists[f"prosody.{pkey}"] = []
        elif key == "features":
            value_lists["features"] = []
        else:
            value_lists[key] = []

    # Fill lists
    for result in results:
        for key in keys:
            if key == "prosody" and "prosody" in result and isinstance(result["prosody"], dict):
                for pkey, pval in result["prosody"].items():
                    if isinstance(pval, (int, float)):
                        value_lists[f"prosody.{pkey}"].append(pval)
            elif key == "features" and "features" in result and isinstance(result["features"], (list, tuple)):
                value_lists["features"].append(result["features"])
            elif key in result and isinstance(result[key], (int, float)):
                value_lists[key].append(result[key])

    # Calculate means for normal keys and prosody
    for key, vals in value_lists.items():
        if key == "features" and vals:
            # Transpose list of lists to get per-index values
            features_arr = list(zip(*vals))
            for i, feature_vals in enumerate(features_arr):
                mean_results[f"mean_features_{i}"] = sum(feature_vals) / len(feature_vals)
                mean_results[f"all_features_{i}"] = list(feature_vals)
            mean_results["all_features"] = vals
        elif vals:
            mean_results[f"mean_{key}"] = sum(vals) / len(vals)
            mean_results[f"all_{key}"] = vals

    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(mean_results, f, indent=4)

    print(f"Mean results saved to {save_path}")


def calculate_prosody_features(audio_path, whisper_model, embedding_model, sentiment_pipeline):
    """
    Given an audio file path, calculate prosody features using the provided models.
    Returns a dictionary of prosody features.
    """
    data = {}
    text = voice_to_text(audio_path, whisper_model)
    prosody = extract_prosody_features(audio_path, whisper_model)
    features = extract_implicit_features(text, embedding_model, sentiment_pipeline, prosody, only_voice_test=True)
    data["prosody"] = prosody
    data["features"] = features
    return data


def generate_and_store_speech(
    dataset,
    base_model,
    processor,
    vocoder,
    get_speaker_embedding,
    get_tone_adapter,
    save_dir,
    base_dir="audios",
    limit=None
):
    """
    Generate baseline and adapter-based speech samples from a dataset,
    save audio files and corresponding metrics in structured directories.
    """
    os.makedirs(base_dir, exist_ok=True)

    for idx, sample in enumerate(dataset if limit is None else dataset[:limit]):
        process_sample(idx, sample, base_model, processor, vocoder, 
                       get_speaker_embedding, get_tone_adapter, save_dir, base_dir)


def process_sample(idx, sample, base_model, processor, vocoder, 
                  get_speaker_embedding, get_tone_adapter, save_dir, base_dir):
    """Process a single dataset sample for speech generation"""
    emotion = sample["major_emotion"]
    text = sample["text"]
    gender = sample.get("gender", "female").lower()
    
    # Get sampling rate
    sr = get_sampling_rate(sample)
    
    # Get inputs for model
    inputs = processor(text=text, return_tensors="pt")
    speaker_embeddings = get_speaker_embedding(gender=gender)
    
    # Extract metrics from sample
    extra_metrics = extract_metrics_from_sample(sample)
    
    # Generate speech for each fold
    for fold in range(1, 4):
        # Generate baseline speech
        generate_speech_type(
            "baseline", idx, fold, text, gender, emotion, "neutral",
            base_model, None, inputs, speaker_embeddings, vocoder, 
            sr, base_dir, extra_metrics
        )
        
        # Generate adapter speech
        adapter_model = get_tone_adapter(base_model, emotion, gender, save_dir=save_dir)
        generate_speech_type(
            "adapter", idx, fold, text, gender, emotion, emotion,
            base_model, adapter_model, inputs, speaker_embeddings, vocoder, 
            sr, base_dir, extra_metrics
        )


def get_sampling_rate(sample):
    """Extract sampling rate from sample or use default"""
    sr = 16000  # Default
    audio_info = sample.get("audio", {})
    audio_path = audio_info.get("path")
    
    if audio_path is None or not os.path.isfile(audio_path):
        arr, sr = audio_info.get("array"), audio_info.get("sampling_rate", 16000)
    else:
        try:
            sr = sf.info(audio_path).samplerate
        except Exception:
            sr = 16000
    return sr


def extract_metrics_from_sample(sample):
    """Extract standard metrics from a sample"""
    return {
        "major_emotion": sample.get("major_emotion"),
        "gender": sample.get("gender"),
        "EmoAct": sample.get("EmoAct"),
        "EmoVal": sample.get("EmoVal"),
        "EmoDom": sample.get("EmoDom"),
        "speaking_rate": sample.get("speaking_rate"),
        "pitch_mean": sample.get("pitch_mean"),
        "pitch_std": sample.get("pitch_std"),
        "rms": sample.get("rms"),
        "relative_db": sample.get("relative_db"),
    }


def generate_speech_type(
    speech_type, idx, fold, text, gender, source_emotion, output_emotion,
    base_model, adapter_model, inputs, speaker_embeddings, vocoder,
    sr, base_dir, extra_metrics
):
    """Generate a specific type of speech (baseline or adapter)"""
    print(f"[{idx}] 🎤 Generating {speech_type} speech for [{source_emotion}] [{gender}], fold {fold}...")
    
    # Use the appropriate model for generation
    model_to_use = adapter_model if adapter_model else base_model
    speech = model_to_use.generate_speech(
        inputs["input_ids"],
        speaker_embeddings,
        vocoder=vocoder
    )

    # Setup directories and paths
    output_dir = os.path.join(base_dir, speech_type, gender)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{idx}_{fold}_{speech_type}.wav")
    
    # Save the audio file
    sf.write(output_path, speech.numpy(), samplerate=sr)
    print(f"✅ Saved {speech_type} speech at {output_path}")
    
    # Calculate prosody features
    prosody_data = calculate_prosody_features(output_path, whisper_model, embedding_model, sentiment_pipeline)
    
    # Create and save metrics
    metrics = {
        "id": idx,
        "fold": fold,
        "text": text,
        "emotion": output_emotion,
        "gender": gender,
        "relative_path": os.path.relpath(output_path, start=base_dir),
        "params": {
            "samplerate": sr,
            "length": len(speech.numpy())
        },
        "prosody": prosody_data.get("prosody", {}),
        "features": prosody_data.get("features", []),
        **extra_metrics
    }
    
    metrics_path = os.path.join(output_dir, f"metrics_{idx}_{fold}_{speech_type}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

