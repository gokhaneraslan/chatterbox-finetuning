import os
import re
import random
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from src.model import resize_and_load_t3_weights
from src.utils import setup_logger, trim_silence_with_vad, load_t3_state
from src.config import TrainConfig
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("Chatterbox-Inference")

cfg = TrainConfig()

IS_TURBO = cfg.is_turbo
IS_LORA = cfg.is_lora
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir
MODE_NAME = "Nano" if cfg.is_nano else "Turbo" if IS_TURBO else "Standard"

LORA_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "new_lang_adapter")
FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, cfg.full_ft_filename)  # Full fine-tune only

# Generation params (Turbo/Nano ignore cfg_weight and exaggeration)
if IS_TURBO:
    PARAMS = {
        "temperature": 0.4,
        "repetition_penalty": 1.2,
    }
else:
    PARAMS = {
        "temperature": 0.7,
        "exaggeration": 0.8,
        "cfg_weight": 0.2,
        "repetition_penalty": 1.2,
    }

TEXT_TO_SAY = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."
AUDIO_PROMPT = "./speaker_reference/2.wav"
OUTPUT_FILE = "./output.wav"


# -----------------------------------------------------------------------------
# Engine loaders
# -----------------------------------------------------------------------------

def load_base_engine():
    """Loads the base engine on CPU and verifies it matches the config."""
    EngineClass = ChatterboxTurboTTS if IS_TURBO else ChatterboxTTS
    engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")

    if IS_TURBO:
        if engine.model_label != MODE_NAME:
            raise RuntimeError(f"Config expects {MODE_NAME}, but '{BASE_MODEL_DIR}' contains {engine.model_label}.")
        tok_len = len(engine.tokenizer)
        if tok_len != cfg.new_vocab_size:
            raise ValueError(f"Tokenizer len {tok_len} != new_vocab_size {cfg.new_vocab_size}. Check the output of setup.py.")

    return engine


def load_finetuned_engine_lora(device):
    """Loads the engine and restores a LoRA-finetuned T3 (with resized vocab)."""
    from peft import PeftModel

    logger.info(f"Loading {MODE_NAME} [LoRA] from: {BASE_MODEL_DIR}")

    if not os.path.exists(LORA_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Adapter path not found: {LORA_CHECKPOINT_PATH}")

    # 1. Load base model temporarily to grab pretrained weights + config
    temp_engine = load_base_engine()
    pretrained_state = temp_engine.t3.state_dict()
    new_t3_config = temp_engine.t3.hp

    # 2. Build new T3 with expanded vocab
    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False

    new_t3 = T3(hp=new_t3_config)

    # 3. Transfer & resize pretrained weights
    logger.info("Resizing base weights to fit new vocab...")
    new_t3 = resize_and_load_t3_weights(new_t3, pretrained_state)

    del temp_engine
    del pretrained_state

    # 4. Turbo/Nano: remove wte to match the training architecture
    if IS_TURBO and hasattr(new_t3.tfmr, "wte"):
        logger.info(f"{MODE_NAME} Mode: Removing 'wte' layer.")
        del new_t3.tfmr.wte

    # 5. Fresh engine + rebuilt T3
    tts_engine = load_base_engine()
    tts_engine.t3 = new_t3

    # 6. LoRA adapters + trained embeddings (modules_to_save)
    logger.info(f"Loading LoRA adapters & embeddings from: {LORA_CHECKPOINT_PATH}")
    tts_engine.t3 = PeftModel.from_pretrained(
        tts_engine.t3,
        LORA_CHECKPOINT_PATH,
        is_trainable=False,
    )

    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device

    return tts_engine


def load_finetuned_engine_full(device):
    """Loads the engine and restores a fully fine-tuned T3 from safetensors."""

    logger.info(f"Loading {MODE_NAME} [Full Fine-Tune] from: {BASE_MODEL_DIR}")

    if not os.path.exists(FINETUNED_WEIGHTS):
        raise FileNotFoundError(f"Fine-tuned file not found: {FINETUNED_WEIGHTS}")

    tts_engine = load_base_engine()

    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size

    new_t3 = T3(hp=t3_config)

    if IS_TURBO and hasattr(new_t3.tfmr, "wte"):
        logger.info(f"{MODE_NAME} Mode: Removing 'wte' layer.")
        del new_t3.tfmr.wte

    logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
    state_dict = load_file(FINETUNED_WEIGHTS, device="cpu")
    load_t3_state(new_t3, state_dict)
    logger.info("Fine-tuned weights loaded successfully.")

    tts_engine.t3 = new_t3
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device

    return tts_engine


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def generate_sentence_audio(engine, text, prompt_path, **kwargs):
    """Generates audio for a single sentence and trims silence."""
    try:
        wav_tensor = engine.generate(text=text, audio_prompt_path=prompt_path, **kwargs)
        if isinstance(wav_tensor, tuple):
            wav_tensor = wav_tensor[0]
        wav_np = wav_tensor.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
        return engine.sr, trimmed_wav
    except Exception as e:
        logger.error(f"Error generating sentence '{text[:30]}...': {e}")
        return 24000, np.zeros(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference running on: {device}")
    logger.info(f"Mode: {MODE_NAME} | Strategy: {'LoRA' if IS_LORA else 'Full Fine-Tune'}")

    if IS_LORA:
        engine = load_finetuned_engine_lora(device)
    else:
        engine = load_finetuned_engine_full(device)

    sentences = re.split(r'(?<=[.?!])\s+', TEXT_TO_SAY.strip())
    sentences = [s for s in sentences if s.strip()]

    logger.info(f"Found {len(sentences)} sentences to synthesize.")

    all_chunks = []
    sample_rate = 24000

    set_seed(42)

    for i, sent in enumerate(sentences):
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        sr, audio_chunk = generate_sentence_audio(engine, sent, AUDIO_PROMPT, **PARAMS)

        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(OUTPUT_FILE, final_audio, sample_rate)
        logger.info(f"Result saved to: {OUTPUT_FILE}")
    else:
        logger.error("No audio was generated.")


if __name__ == "__main__":
    main()