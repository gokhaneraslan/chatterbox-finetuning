import os
import glob
import torch
import torchaudio
from tqdm import tqdm

from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger, check_preprocess_meta, write_preprocess_meta
from src.config import TrainConfig


logger = setup_logger(__name__)


def preprocess_dataset_file_based(config, tts_engine: ChatterboxTTS):
    """
    Reads .wav and .txt file pairs from a folder, processes them, and saves them as .pt.
    Structure: ID.wav (audio) + ID.txt (text)
    """

    os.makedirs(config.preprocessed_dir, exist_ok=True)

    # Turbo/Nano: tokenizer must match new_vocab_size, and the folder must not hold data from another tokenizer
    if config.is_turbo:
        tok_len = len(tts_engine.tokenizer)
        if tok_len != config.new_vocab_size:
            raise ValueError(f"Tokenizer len {tok_len} != new_vocab_size {config.new_vocab_size}. Check the output of setup.py.")
        check_preprocess_meta(config.preprocessed_dir, tok_len)
        write_preprocess_meta(config.preprocessed_dir, tok_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)
    tts_engine.ve.eval()
    tts_engine.s3gen.eval()

    wav_files = sorted(glob.glob(os.path.join(config.wav_dir, "*.wav")))

    if len(wav_files) == 0:
        logger.error(f"No .wav files found in folder '{config.wav_dir}'!")
        return

    logger.info(f"Processing dataset... Found audio files: {len(wav_files)}")

    success_count = 0
    skipped_count = 0

    SPEECH_STOP_ID = getattr(tts_engine.t3.hp, 'stop_speech_token', 6562)
    for wav_path in tqdm(wav_files, desc="Preprocessing"):

        file_id = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            txt_path = os.path.join(config.wav_dir, f"{file_id}.txt")

            if not os.path.exists(txt_path):
                logger.warning(f"Text file not found, skipping: {file_id}")
                skipped_count += 1
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()

            if not raw_text:
                skipped_count += 1
                continue

            wav, sr = torchaudio.load(wav_path)

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != S3_SR:
                resampler = torchaudio.transforms.Resample(sr, S3_SR)
                wav = resampler(wav)

            wav = wav.to(device)

            with torch.no_grad():

                wav_np = wav.cpu().squeeze().numpy()

                spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
                speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()

                s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
                raw_speech_tokens = s_tokens.squeeze().cpu()

                stop_speech_tensor = torch.tensor([SPEECH_STOP_ID], dtype=raw_speech_tokens.dtype)
                speech_tokens = torch.cat([raw_speech_tokens, stop_speech_tensor], dim=0)

                prompt_samples = int(config.prompt_duration * S3_SR)
                if wav.shape[1] < prompt_samples:
                    prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                else:
                    prompt_wav = wav[:, :prompt_samples]

                p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
                prompt_tokens = p_tokens.squeeze().cpu()

            clean_text = punc_norm(raw_text)

            if config.is_turbo:
                token_output = tts_engine.tokenizer(clean_text, return_tensors="pt")
                raw_text_tokens = token_output.input_ids[0].cpu()

                if tts_engine.tokenizer.eos_token_id is not None:
                    text_eos = torch.tensor([tts_engine.tokenizer.eos_token_id], dtype=raw_text_tokens.dtype)
                    text_tokens = torch.cat([raw_text_tokens, text_eos], dim=0)
                else:
                    text_tokens = raw_text_tokens

            else:
                text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()

            save_path = os.path.join(config.preprocessed_dir, f"{file_id}.pt")

            torch.save({
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens,
            }, save_path)

            success_count += 1

        except Exception as e:
            logger.error(f"Error ({file_id}): {e}")
            continue

    logger.info(f"Preprocessing completed! Success: {success_count}/{len(wav_files)} | Skipped: {skipped_count}")


if __name__ == "__main__":

    cfg = TrainConfig()

    EngineClass = ChatterboxTurboTTS if cfg.is_turbo else ChatterboxTTS

    logger.info(f"{EngineClass.__name__} engine starting...")
    tts_engine = EngineClass.from_local(cfg.model_dir, device="cpu")

    preprocess_dataset_file_based(cfg, tts_engine)