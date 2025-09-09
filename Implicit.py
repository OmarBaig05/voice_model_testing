import librosa
import parselmouth
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader
from datasets import Dataset
import logging
import torch
import json
from peft import PeftModel



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # enable debug logs for detailed tracing

def voice_to_text(audio_path: str, whisper_model) -> str:
    try:
        result = whisper_model.transcribe(audio_path, language="en")
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Voice-to-text conversion failed: {e}")
        return ""

def extract_prosody_features(audio_path: str, whisper_model) -> Dict[str, float]:
    """
    Extract and normalize prosody features from audio:
    - mean pitch
    - pitch std
    - RMS energy
    - speaking rate

    Returns normalized values.
    """
    try:
        logger.debug(f"extract_prosody_features called with audio_path={audio_path}")

        # Preserve original sample rate to avoid resampling artifacts
        y, sr = librosa.load(audio_path, sr=None)
        logger.debug(f"librosa.load -> y.shape={getattr(y, 'shape', None)}, sr={sr}")

        try:
            snd = parselmouth.Sound(audio_path)
            logger.debug("parselmouth.Sound loaded successfully")
        except Exception as e:
            logger.exception(f"parselmouth failed to load sound: {e}")
            raise

        # Pitch extraction
        pitch = snd.to_pitch()
        # pitch.selected_array['frequency'] may contain NaNs; handle safely
        try:
            raw_pitch = pitch.selected_array['frequency']
            logger.debug(f"Raw pitch array length: {len(raw_pitch)}")
        except Exception as e:
            logger.exception(f"Failed to read pitch.selected_array['frequency']: {e}")
            raw_pitch = np.array([])

        # Filter out non-finite values and unrealistic pitch values
        if raw_pitch.size > 0:
            finite_mask = np.isfinite(raw_pitch)
            pitch_values = raw_pitch[finite_mask]
            logger.debug(f"Pitch finite count: {pitch_values.size}")
            if pitch_values.size > 0:
                pitch_values = pitch_values[(pitch_values > 50) & (pitch_values < 500)]
                logger.debug(f"Pitch after range filter (50-500 Hz) count: {pitch_values.size}")
            else:
                logger.debug("No finite pitch values found after filtering NaNs/Infs")
        else:
            pitch_values = np.array([])
            logger.debug("Raw pitch array is empty")

        mean_pitch = np.mean(pitch_values) if pitch_values.size > 0 else 0.0
        std_pitch = np.std(pitch_values) if pitch_values.size > 0 else 0.0
        logger.debug(f"mean_pitch(raw)={mean_pitch}, std_pitch(raw)={std_pitch}")

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)
        rms = rms[0] if rms.size > 0 else np.array([])
        logger.debug(f"RMS array length: {rms.size}, sample values (first 5): {rms[:5] if rms.size>0 else []}")
        mean_rms = float(np.mean(rms)) if rms.size > 0 else 0.0
        logger.debug(f"mean_rms(raw)={mean_rms}")

        # Duration and speaking rate
        duration = librosa.get_duration(y=y, sr=sr)
        logger.debug(f"duration={duration} seconds")
        text = voice_to_text(audio_path, whisper_model=whisper_model)
        logger.debug(f"voice_to_text returned text length={len(text)}")
        word_count = len(text.split()) if text else 0
        speaking_rate = word_count / duration if duration > 0 else 0.0
        logger.debug(f"word_count={word_count}, speaking_rate(raw)={speaking_rate}")

        # Normalization (approx ranges; tweak if needed)
        mean_pitch_norm = (mean_pitch - 100) / 100     # assuming 100–300 Hz range
        std_pitch_norm = std_pitch / 50                # assuming max 50 Hz std
        mean_rms_norm = (mean_rms - 0.01) / 0.02        # assuming range 0.01–0.03
        speaking_rate_norm = (speaking_rate - 2) / 2    # assuming normal 2 wps

        logger.debug({
            "mean_pitch": mean_pitch,
            "std_pitch": std_pitch,
            "mean_rms": mean_rms,
            "speaking_rate": speaking_rate,
            "mean_pitch_norm": mean_pitch_norm,
            "std_pitch_norm": std_pitch_norm,
            "mean_rms_norm": mean_rms_norm,
            "speaking_rate_norm": speaking_rate_norm
        })

        return {
            "mean_pitch": float(mean_pitch_norm),
            "std_pitch": float(std_pitch_norm),
            "mean_rms": float(mean_rms_norm),
            "speaking_rate": float(speaking_rate_norm)
        }

    except Exception as e:
        logger.exception(f"Prosody feature extraction failed: {e}")
        return {
            "mean_pitch": 0.0,
            "std_pitch": 0.0,
            "mean_rms": 0.0,
            "speaking_rate": 0.0
        }

def extract_implicit_features(text: str, embedding_model, sentiment_pipeline, prosody: Dict[str, float], only_voice_test = False) -> torch.Tensor:
    """
    Merge sentiment + embedding + prosody features into a single tensor.
    """
    try:
        logger.info(f"Processed text: {text}")

        # Sentiment
        sentiment = sentiment_pipeline(text)[0]
        sentiment_score = sentiment["score"]
        sentiment_label = sentiment["label"]
        label_to_id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
        sentiment_label_id = label_to_id.get(sentiment_label.upper(), 1)

        # Embedding
        embedding = embedding_model.encode([text])[0]

        # Merge all
        prosody_values = [
            prosody.get("mean_pitch", 0.0),
            prosody.get("std_pitch", 0.0),
            prosody.get("mean_rms", 0.0),
            prosody.get("speaking_rate", 0.0),
        ]

        if only_voice_test == True:
            print("inside only voice test")
            merged_embedding = torch.tensor(
            [sentiment_score, sentiment_label_id] + prosody_values,
            dtype=torch.float32
        )
            return merged_embedding

        merged_embedding = torch.tensor(
            [sentiment_score, sentiment_label_id] + prosody_values + list(embedding),
            dtype=torch.float32
        )
        return merged_embedding

    except Exception as e:
        logger.error(f"Implicit feature extraction failed: {e}")
        return torch.tensor([], dtype=torch.float32)

def update_lora_weights(texts: List[str], model, output_dir: str, embedding_model, sentiment_pipeline, prosody_list: List[Dict[str, float]], user_id: str = None, adapter_type: str = "local"):

    try:
        # Attach adapter to the base model
        model_peft = PeftModel.from_pretrained(model, output_dir.replace("\\", "/"))

        dataset = Dataset.from_dict({"text": texts})
        dataloader = DataLoader(dataset, batch_size=1)

        for i, batch in enumerate(dataloader):
            text = batch["text"][0]
            prosody = prosody_list[i] if i < len(prosody_list) else {}
            
            merged_features = extract_implicit_features(
                text, embedding_model, sentiment_pipeline, prosody
            )
            logger.info(f"Merged feature vector shape: {merged_features.shape}")

            try:
                for name, param in model_peft.named_parameters():
                    if "lora" in name and param.requires_grad:
                        if adapter_type == "local" or user_id is None:
                            print("inside implicit files local adapter weight update")
                            param.data += 0.001 * merged_features.mean()
                        elif adapter_type == "global" and user_id is not None:
                            print("inside implicit files global adapter weight update")
                            param.data += 0.0005 * merged_features.mean()
            except Exception as e:
                logger.error(f"LoRA update failed: {e}")

        # Save only adapter (not the whole model)
        model_peft.save_pretrained(output_dir.replace("\\", "/"), safe_serialization=True)
        logger.info(f"LoRA {adapter_type} adapter updated and saved to {output_dir}")

    except Exception as e:
        logger.error(f"LoRA weight update loop failed: {e}")

def Implicit_main(input_data, whisper_model, model, embedding_model, sentiment_pipeline, 
                 voice_metadata: str = None, user_id: str = None, adapter_manager = None):
    try:
        # Convert single string to list if needed
        texts = [input_data] if isinstance(input_data, str) else input_data
        
        if not texts or not all(isinstance(t, str) for t in texts):
            logger.error(f"Invalid input: expected string or list of strings, got {type(input_data)}")
            return
            
        logger.info(f"Processing {len(texts)} text entries")
        
        # Prepare prosody features list (one per text)
        prosody_list = []
        
        # Process each text entry
        for i, text in enumerate(texts):
            # Check if we have voice recording for this text
            has_voice = False
            
            if voice_metadata:
                # voice_metadata is now a string (the path), not a dict
                voice_path = voice_metadata
                logger.info(f"Using provided voice recording for text '{text[:30]}...': {voice_path}")
                try:
                    prosody = extract_prosody_features(voice_path, whisper_model)
                    prosody_list.append(prosody)
                    logger.info(f"Extracted prosody features: {prosody}")
                    has_voice = True
                except Exception as e:
                    logger.error(f"Failed to extract prosody features from {voice_path}: {e}")
                    # Add default prosody as fallback
                    prosody_list.append({
                        "mean_pitch": 0.0,
                        "std_pitch": 0.0,
                        "mean_rms": 0.0,
                        "speaking_rate": 0.0
                    })
            
            # If no voice recording found or processing failed, add default prosody
            if not has_voice:
                logger.info(f"No voice data for text '{text[:30]}...', using default prosody")
                prosody_list.append({
                    "mean_pitch": 0.0,
                    "std_pitch": 0.0,
                    "mean_rms": 0.0,
                    "speaking_rate": 0.0
                })
        
        # Ensure we have prosody features for each text
        if len(prosody_list) != len(texts):
            logger.warning(f"Mismatch between texts ({len(texts)}) and prosody features ({len(prosody_list)})")
            # Extend with defaults if needed
            while len(prosody_list) < len(texts):
                prosody_list.append({
                    "mean_pitch": 0.0,
                    "std_pitch": 0.0,
                    "mean_rms": 0.0,
                    "speaking_rate": 0.0
                })
        
        # If we have an adapter manager and user_id, use it
        if adapter_manager and user_id:
            adapter_paths = adapter_manager.get_user_adapter_paths(user_id)
            print("@ User Adapter Path: ",adapter_paths)
            
            # Update the local adapter (implicit features only affect local)
            update_lora_weights(
                texts, model, adapter_paths["local"], 
                embedding_model, sentiment_pipeline, prosody_list,
                user_id=user_id, adapter_type="local"
            )
            
            logger.info(f"Updated local adapter for user {user_id} with implicit features")
        else:
            logger.info("Cannot do LoRA update, adapter_manager or user_id is not passed")

    except Exception as e:
        logger.error(f"Implicit_main function failed: {e}", exc_info=True)


if __name__ == "__main__":
    pass
