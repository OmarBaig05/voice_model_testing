import os
import json
import torch
import soundfile as sf
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from peft import PeftModel, LoraConfig
from huggingface_hub import hf_hub_download
import numpy as np
import zipfile

# -----------------------------
# Load base SpeechT5 models
# -----------------------------
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
base_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


# -----------------------------
# Create/load LoRA adapter
# -----------------------------
def get_tone_adapter(model, tone: str, save_dir="models", r=8, alpha=16, dropout=0.05):
    """
    Load LoRA adapter if it exists, otherwise create and save it.
    Gender-specific saving inside tone dir.
    """
    # top-level directory
    tone_dir = os.path.join(save_dir, tone)
    os.makedirs(tone_dir, exist_ok=True)

    # nested path where PEFT actually saves (tone_dir/tone/adapter_config.json)
    nested_dir = os.path.join(tone_dir, tone)

    if os.path.exists(os.path.join(nested_dir, "adapter_config.json")):
        print(f"🔄 Loading existing adapter for tone '{tone}' from {nested_dir}")
        peft_model = PeftModel.from_pretrained(model, nested_dir, adapter_name=tone)
    else:
        print(f"✨ Creating new adapter for tone '{tone}'...")
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        peft_model = PeftModel(model, config, adapter_name=tone)
        peft_model.save_pretrained(tone_dir, safe_serialization=True)  # PEFT will create tone_dir/tone/
        print(f"✅ Adapter for tone '{tone}' saved at: {nested_dir}")

    return peft_model


# -----------------------------
# Speaker embedding loader
# -----------------------------
def get_speaker_embedding(gender: str = "female", cache_dir="speaker_embeddings"):
    """
    Load cached speaker embedding if available, otherwise fetch and cache it.
    Gender: 'male' or 'female'
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"speaker_embedding_{gender.lower()}.npy")

    if os.path.exists(cache_path):
        print(f"🔄 Loading cached {gender} speaker embedding from {cache_path}")
        xvector = np.load(cache_path)
    else:
        print(f"✨ Fetching new {gender} speaker embedding...")
        try:
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

            if gender.lower() == "female":
                sample = next(x for x in embeddings_dataset if "cmu_us_slt_" in x["filename"])
                print("🎙 Using female speaker: slt (Susan)")
            else:
                sample = next(x for x in embeddings_dataset if "cmu_us_bdl_" in x["filename"])
                print("🎙 Using male speaker: bdl")

            xvector = sample["xvector"]

        except Exception:
            print("⚠️ Falling back to manual download...")
            zip_path = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset")
            extract_dir = "cmu-arctic-xvectors"
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            npy_files = [os.path.join(root, f)
                         for root, _, files in os.walk(extract_dir) for f in files if f.endswith(".npy")]
            if not npy_files:
                raise RuntimeError("No .npy files found in speaker embeddings.")

            if gender.lower() == "female":
                female_files = [f for f in npy_files if "cmu_us_slt" in f or "cmu_us_clb" in f]
                if not female_files:
                    raise RuntimeError("No female speaker embeddings found.")
                xvector = np.load(female_files[0])
                print(f"🎙 Using female speaker from file: {os.path.basename(female_files[0])}")
            else:
                male_files = [f for f in npy_files if "cmu_us_bdl" in f or "cmu_us_rms" in f]
                if not male_files:
                    raise RuntimeError("No male speaker embeddings found.")
                xvector = np.load(male_files[0])
                print(f"🎙 Using male speaker from file: {os.path.basename(male_files[0])}")

        np.save(cache_path, xvector)
        print(f"✅ {gender.capitalize()} speaker embedding cached at {cache_path}")

    return torch.tensor(xvector).unsqueeze(0)


# -----------------------------
# Adapter weight updater
# -----------------------------
def update_emotion_adapters(model, dataset: list, save_dir: str):
    texts = [item["text"] for item in dataset]
    emotions = [item["major_emotion"] for item in dataset]
    genders = [item["gender"] for item in dataset]

    # ✅ Convert features back into torch tensors
    features_list = [
        torch.tensor(item["features"], dtype=torch.float32)
        if isinstance(item["features"], (list, tuple)) else item["features"]
        for item in dataset
    ]

    ds = Dataset.from_dict({"text": texts, "emotion": emotions, "gender": genders})
    dataloader = DataLoader(ds, batch_size=1)

    for i, batch in enumerate(dataloader):
        emotion = batch["emotion"][0]
        gender = batch["gender"][0]
        features = features_list[i]

        adapter_dir = os.path.join(save_dir, gender.lower(), emotion, emotion)
        model_peft = PeftModel.from_pretrained(model, adapter_dir, adapter_name=emotion)

        # ✅ Unfreeze LoRA parameters
        for name, param in model_peft.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        print(f"🔄 Updating adapter for [{emotion}]...")

        for name, param in model_peft.named_parameters():
            if "lora" in name and param.requires_grad:
                before = param.data.clone().detach().cpu()
                param.data += 0.01 * features.mean()
                after = param.data.clone().detach().cpu()

                print(f"➡️ {name}: Δmean={after.mean() - before.mean():.8f}")

        # Save updated adapter
        model_peft.save_pretrained(adapter_dir, safe_serialization=True)

# -----------------------------
# Main Example
# -----------------------------
if __name__ == "__main__":
    # Load dataset and convert features back
    dataset_path = "results.json"   # <-- put your dataset JSON file here
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    dataset = dataset[:10]

    # (Already handled above inside update_emotion_adapters)
    save_dir = "models"

    # Create adapters first (one per emotion)
    unique_emotions = set(item["major_emotion"] for item in dataset)
    for emotion in unique_emotions:
        _ = get_tone_adapter(base_model, emotion, save_dir=save_dir)

    # Update weights with features
    update_emotion_adapters(base_model, dataset, save_dir=save_dir)
