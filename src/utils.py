import json
import logging
import os
import sys
import torch
import torchaudio
import numpy as np


def setup_logger(name: str, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


_logger = setup_logger("Utils")

_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None


def load_vad_model():
    """Lazy loads the Silero VAD model."""

    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS

    if _VAD_MODEL is not None:
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS

    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )

        _GET_SPEECH_TIMESTAMPS = utils[0]
        _VAD_MODEL = model

        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS

    except Exception as e:
        print(f"Error loading VAD: {e}")
        return None, None


def trim_silence_with_vad(audio_waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trims silence/noise from the end of the audio using Silero VAD."""

    vad_model, get_timestamps = load_vad_model()
    if vad_model is None:
        return audio_waveform

    VAD_SR = 16000
    audio_tensor = torch.from_numpy(audio_waveform).float()

    # Resample for VAD if necessary
    if sample_rate != VAD_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SR)
        vad_input = resampler(audio_tensor)
    else:
        vad_input = audio_tensor

    try:
        speech_timestamps = get_timestamps(vad_input, vad_model, sampling_rate=VAD_SR)

        if not speech_timestamps:
            return audio_waveform

        # End of the last speech chunk, scaled back to the original sample rate
        last_speech_end_vad = speech_timestamps[-1]['end']
        cut_point = int(last_speech_end_vad * (sample_rate / VAD_SR))

        return audio_waveform[:cut_point]

    except Exception as e:
        print(f"VAD trimming failed: {e}")
        return audio_waveform


def check_pretrained_models(model_dir="pretrained_models", mode="chatterbox"):
    """Checks that the files loaded by from_local() exist in model_dir."""

    if mode in ("chatterbox_turbo", "chatterbox_nano"):
        t3_file = "t3_nano_v1.safetensors" if mode == "chatterbox_nano" else "t3_turbo_v1.safetensors"
        required_files = [
            t3_file,
            "ve.safetensors",
            "s3gen_meanflow.safetensors",
            "conds.pt",
            "vocab.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "merges.txt",
            "grapheme_mtl_merged_expanded_v1.json"
        ]

    else:
        required_files = [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "conds.pt",
            "tokenizer.json"
        ]

    missing_files = []

    if not os.path.exists(model_dir):
        print(f"\nERROR: '{model_dir}' folder doesn't exist!")
        missing_files = required_files

    else:
        for filename in required_files:
            if not os.path.exists(os.path.join(model_dir, filename)):
                missing_files.append(filename)

    if missing_files:
        print("\n" + "!" * 60)
        print("ATTENTION: The following model files could not be found:")
        for f in missing_files:
            print(f"   - {f}")

        print("\nPlease run the following command to download the models:")
        print(" python setup.py")
        print("!" * 60 + "\n")
        return False

    print(f"All necessary models are available under '{model_dir}'.")
    return True


def load_t3_state(t3, state_dict):
    """
    Loads a T3 checkpoint after stripping wrapper prefixes ('module.', 'model.', 't3.').
    Raises if any weight is missing (the deleted tfmr.wte is ignored).
    """
    clean = {}
    for k, v in state_dict.items():
        for prefix in ("module.", "model.", "t3."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        if not k.startswith(("s3gen.", "ve.", "tokenizer.")):
            clean[k] = v

    missing, unexpected = t3.load_state_dict(clean, strict=False)
    missing = [k for k in missing if "wte" not in k]
    if missing:
        raise RuntimeError(f"Missing T3 keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        _logger.warning(f"Unexpected T3 keys ({len(unexpected)}): {unexpected[:5]}")
    return t3


META_NAME = "preprocess_meta.json"


def write_preprocess_meta(directory, tokenizer_len):
    """Records the tokenizer size used to build the preprocessed data."""
    with open(os.path.join(directory, META_NAME), "w") as f:
        json.dump({"tokenizer_len": tokenizer_len}, f)


def check_preprocess_meta(directory, expected_len):
    """Fails fast if the preprocessed data was built with a different tokenizer."""
    path = os.path.join(directory, META_NAME)

    if not os.path.exists(path):
        if any(f.endswith(".pt") for f in os.listdir(directory)):
            _logger.warning(f"'{META_NAME}' not found in {directory}; cannot verify the tokenizer. Re-run preprocessing to create it.")
        return

    with open(path) as f:
        found = json.load(f)["tokenizer_len"]

    if found != expected_len:
        raise ValueError(f"Data in '{directory}' was built with tokenizer len {found}, expected {expected_len}. "
                         f"Delete the folder and re-run preprocessing.")