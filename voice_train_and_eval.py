import logging
from Implicit import extract_prosody_features, extract_implicit_features
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from datetime import datetime

logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")



"""
tacotron2_prosody_adapter.py

Wraps SpeechBrain Tacotron2 with a small ProsodyAdapter (LoRA-like).
Provides functions to update adapter weights using prosody features only,
and to save/load the adapter in the adapter_manager-friendly format.

Assumptions:
- SpeechBrain tacotron2 usage as in your provided snippet.
- Prosody vectors are dicts with keys: mean_pitch, std_pitch, mean_rms, speaking_rate (float).
- torch and speechbrain are installed.
"""

# -------------------------
# Prosody Adapter (LoRA-like)
# -------------------------
class ProsodyAdapter(nn.Module):
    """
    A small adapter that maps prosody features to additive adjustments applied
    to encoder outputs. Implemented as a low-rank bottleneck:
        prosody_vector -> Linear(d_p -> r) -> activation -> Linear(r -> encoder_dim)
    The adapter output is broadcast and added to the encoder outputs.
    """
    def __init__(self, prosody_dim: int = 4, encoder_dim: int = 512, rank: int = 32):
        super().__init__()
        self.prosody_dim = prosody_dim
        self.encoder_dim = encoder_dim
        self.rank = rank

        # low-rank down & up
        self.down = nn.Linear(prosody_dim, rank, bias=False)
        self.act = nn.ReLU()
        self.up = nn.Linear(rank, encoder_dim, bias=False)

        # small scaling parameter (learnable scalar) to control magnitude
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, encoder_outputs: torch.Tensor, prosody_vector: torch.Tensor):
        """
        encoder_outputs: (batch, seq_len, encoder_dim)
        prosody_vector: (batch, prosody_dim)
        """
        # compute adapter vector per batch: (batch, encoder_dim)
        z = self.down(prosody_vector)
        z = self.act(z)
        z = self.up(z)  # (batch, encoder_dim)
        z = z * self.scale

        # broadcast add over time dimension
        z = z.unsqueeze(1)  # (batch, 1, encoder_dim)
        return encoder_outputs + z  # (batch, seq_len, encoder_dim)

    def get_state_dict(self):
        return self.state_dict()

    def load_state_dict_safe(self, sd: Dict):
        self.load_state_dict(sd)


# -------------------------
# Wrapper around Tacotron2
# -------------------------
class Tacotron2WithAdapter:
    def __init__(self, tacotron_source="speechbrain/tts-tacotron2-ljspeech",
                 hifigan_source="speechbrain/tts-hifigan-ljspeech",
                 savedir_tts="tmpdir_tts",
                 savedir_vocoder="tmpdir_vocoder",
                 adapter: Optional[ProsodyAdapter] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load base models
        self.tacotron2 = Tacotron2.from_hparams(source=tacotron_source, savedir=savedir_tts)
        self.hifi_gan = HIFIGAN.from_hparams(source=hifigan_source, savedir=savedir_vocoder)
        # ensure models on device (speechbrain modules manage their own devices; we will move adapter)
        self.adapter = adapter
        if self.adapter is None:
            # detect encoder dim from tacotron2 internals; best-effort default 512
            encoder_dim = getattr(self.tacotron2, "encoder", None)
            # use default 512 if detection not simple
            self.adapter = ProsodyAdapter(prosody_dim=4, encoder_dim=512, rank=32)
        self.adapter.to(self.device)

        # Monkey patch encode_text to inject adapter adjustments.
        # We'll call original encode_text implementation but intercept encoder outputs.
        # Since SpeechBrain Tacotron2 API doesn't expose internal tensors easily,
        # we instead use encode_text as-is and then apply adapter by shifting mel targets BEFORE vocoder.
        # Note: This is a conservative integration approach: we add prosody perturbation on encoder-level
        # by re-implementing a small forward path where possible. If deeper integration is required,
        # modify internals of SpeechBrain Tacotron2 pipeline.
        # For now, we implement a wrapper around encode_text results: we don't have direct encoder outputs,
        # but Tacotron2.encode_text returns mel_output (batch, time, mel_dim). We'll map adapter output
        # into a **learned linear projection** from encoder_dim -> mel_dim and add to mel_output.
        self.mel_projection = nn.Linear(self.adapter.encoder_dim, self.tacotron2.hparams.n_mels).to(self.device)

    def prosody_vector_from_dict(self, prosody: Dict[str, float]) -> torch.Tensor:
        # order must be fixed and consistent: [mean_pitch, std_pitch, mean_rms, speaking_rate]
        vec = torch.tensor([
            prosody.get("mean_pitch", 0.0),
            prosody.get("std_pitch", 0.0),
            prosody.get("mean_rms", 0.0),
            prosody.get("speaking_rate", 0.0)
        ], dtype=torch.float32, device=self.device)
        return vec.unsqueeze(0)  # (1, 4)

    def synthesize(self, text: str, prosody: Dict[str, float], save_wav_path: Optional[str] = None, sample_rate: int = 22050):
        """
        text -> mel using tacotron2 -> add adapter-inferred mel perturbation -> vocoder -> waveform
        """
        # 1) base mel generation
        mel_output, mel_length, alignment = self.tacotron2.encode_text(text)
        # mel_output shape: (batch=1, time, n_mels)
        # Move to device
        mel_output = mel_output.to(self.device)

        # 2) compute adapter output
        prosody_vec = self.prosody_vector_from_dict(prosody)  # (1, 4)
        # produce adapter vector in encoder_dim space
        # NOTE: our adapter expects encoder outputs shape (1, seq_len, encoder_dim).
        # We'll create a dummy encoder-like tensor of zeros and forward it to obtain a (1, seq_len, encoder_dim)
        seq_len = mel_output.size(1)
        dummy_enc = torch.zeros((1, seq_len, self.adapter.encoder_dim), device=self.device)
        adapted = self.adapter(dummy_enc, prosody_vec)  # (1, seq_len, encoder_dim)
        # project to mel space
        mel_delta = self.mel_projection(adapted)  # (1, seq_len, n_mels)
        # Add to base mel
        mel_adapted = mel_output + mel_delta

        # 3) vocoder -> waveform
        waveforms = self.hifi_gan.decode_batch(mel_adapted)
        wave = waveforms.squeeze(1).cpu()

        if save_wav_path:
            os.makedirs(os.path.dirname(save_wav_path) or ".", exist_ok=True)
            torchaudio.save(save_wav_path, wave, sample_rate)

        return mel_adapted.detach().cpu(), wave  # return mel and waveform tensor

    # -------------------------
    # Adapter save/load in adapter_manager format
    # -------------------------
    def save_adapter(self, out_dir: str):
        """
        Save adapter state dict and adapter_config.json into out_dir.
        Files produced:
          - adapter_model.safetensors  (actually a PyTorch state_dict saved)
          - adapter_config.json
        This matches AdapterManager expectation.
        """
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "adapter_model.safetensors")
        config_path = os.path.join(out_dir, "adapter_config.json")

        # merge adapter + mel_projection params
        sd = {
            "adapter": self.adapter.state_dict(),
            "mel_projection": self.mel_projection.state_dict(),
            "meta": {
                "prosody_dim": self.adapter.prosody_dim,
                "encoder_dim": self.adapter.encoder_dim,
                "rank": self.adapter.rank,
                "saved": datetime.now().isoformat()
            }
        }
        # Use torch.save for portability
        torch.save(sd, model_path)

        # minimal config
        cfg = {
            "type": "prosody_adapter",
            "prosody_keys": ["mean_pitch", "std_pitch", "mean_rms", "speaking_rate"],
            "created": datetime.now().isoformat()
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def load_adapter(self, adapter_dir: str):
        model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        sd = torch.load(model_path, map_location="cpu")
        # load into adapter & mel_projection
        if "adapter" in sd:
            self.adapter.load_state_dict(sd["adapter"])
        if "mel_projection" in sd:
            self.mel_projection.load_state_dict(sd["mel_projection"])


# -------------------------
# Adapter update routine (prosody-only)
# -------------------------
def update_adapter_weights_on_prosody(adapter_dir: str,
                                      prosody_list: List[Dict[str, float]],
                                      lr: float = 1e-3,
                                      method: str = "mean_scale"):
    """
    Load adapter from adapter_dir, update its parameters using prosody_list,
    and save back to adapter_dir (adapter_model.safetensors + adapter_config.json).
    Only adapter and mel_projection params are updated.

    Prosody_list: list aligned with training items; each element is a prosody dict.
    lr: learning rate-like scalar controlling update magnitude.
    method:
      - "mean_scale": compute mean scalar from prosody vector and apply param += lr * mean_scalar
      - "grad_step": perform a small gradient step using a simple MSE objective (requires pseudo-targets) -- not implemented here.
    """
    # Load adapter state
    model_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Adapter file not found at {model_path}")

    sd = torch.load(model_path, map_location="cpu")
    adapter_sd = sd.get("adapter", {})
    melproj_sd = sd.get("mel_projection", {})

    # Convert prosody list to tensors and compute a scalar per sample
    scalars = []
    for p in prosody_list:
        # a simple normalized scalar: weighted combination (tweak as needed)
        mean_pitch = p.get("mean_pitch", 0.0)
        std_pitch = p.get("std_pitch", 0.0)
        mean_rms = p.get("mean_rms", 0.0)
        speaking_rate = p.get("speaking_rate", 0.0)
        # normalization heuristics can be improved; user should preprocess to a reasonable range
        scalar = (mean_pitch * 0.4) + (std_pitch * 0.2) + (mean_rms * 0.3) + (speaking_rate * 0.1)
        scalars.append(float(scalar))

    # take aggregated scalar (mean) across dataset
    if len(scalars) == 0:
        agg = 0.0
    else:
        agg = float(sum(scalars) / len(scalars))

    # Update adapter parameters in place: apply small additive change proportional to agg*lr
    delta = lr * agg

    # We only update LoRA-like matrices in adapter_sd and mel_projection weights
    # adapter_sd keys will typically include 'down.weight', 'up.weight', 'scale' etc.
    for k, v in adapter_sd.items():
        if isinstance(v, torch.Tensor):
            adapter_sd[k] = v + (v * 0.0 + delta)  # add delta (broadcast)
    for k, v in melproj_sd.items():
        if isinstance(v, torch.Tensor):
            melproj_sd[k] = v + (v * 0.0 + delta * 0.5)  # smaller change to mel projection

    # Save back
    new_sd = {
        "adapter": adapter_sd,
        "mel_projection": melproj_sd,
        "meta": {
            "updated": datetime.now().isoformat(),
            "update_delta": delta,
            "method": method
        }
    }
    torch.save(new_sd, model_path)
    # update config timestamp
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    try:
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        cfg["last_updated"] = datetime.now().isoformat()
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

    return delta


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example flow: create wrapper, save adapter, update it with prosody features, synthesize audio
    adapter_dir = "./models/user_123/local_adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    text = transcribe_audio(audio_path)

    prosody = extract_prosody_features(audio_path)

    final_tensor = extract_implicit_features(text, embedding_model, sentiment_pipeline, prosody, only_voice_test=True)

    # 1) create wrapper and save initial adapter
    wrapper = Tacotron2WithAdapter()
    wrapper.save_adapter(adapter_dir)
    print("Initial adapter saved to:", adapter_dir)

    # 2) suppose you have prosody_list from IEMOCAP for multiple utterances
    prosody_list = [
        {"mean_pitch": 120.0, "std_pitch": 18.0, "mean_rms": 0.02, "speaking_rate": 3.0},
        {"mean_pitch": 200.0, "std_pitch": 30.0, "mean_rms": 0.03, "speaking_rate": 4.0},
    ]

    delta = update_adapter_weights_on_prosody(adapter_dir, prosody_list, lr=1e-4)
    print("Adapter updated by delta:", delta)