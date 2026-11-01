import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import setup_logger, check_preprocess_meta


logger = setup_logger(__name__)


class ChatterboxDataset(Dataset):

    def __init__(self, config):
        self.cfg = config
        self.preprocessed_dir = config.preprocessed_dir

        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessing folder not found: {self.preprocessed_dir}.")

        self.files = [f for f in os.listdir(self.preprocessed_dir) if f.endswith(".pt")]

        if len(self.files) == 0:
            raise RuntimeError(f"There are no .pt files in the folder: {self.preprocessed_dir}")

        # Turbo/Nano: the data must come from the same tokenizer as new_vocab_size
        if config.is_turbo:
            check_preprocess_meta(self.preprocessed_dir, config.new_vocab_size)

        logger.info(f"Dataset loaded. Total samples: {len(self.files)}")

        self.sot_token = config.start_text_token
        self.eot_token = config.stop_text_token

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        try:
            pt_path = os.path.join(self.preprocessed_dir, filename)
            data = torch.load(pt_path, weights_only=True)

            text_tokens = data["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[:self.cfg.max_text_len - 2]

            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            speech_tokens = data["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[:self.cfg.max_speech_len]

            speaker_emb = data["speaker_emb"]
            prompt_tokens = data["prompt_tokens"]

            # Randomly drop the speaker conditioning (20%)
            if random.random() < 0.20:
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.zeros(1, dtype=torch.long)

            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens
            }

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None


def data_collator_standart(batch):

    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens
    }


def data_collator_turbo(batch):

    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)

    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_lens = torch.tensor([x["prompt_tokens"].shape[0] for x in batch], dtype=torch.long)

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
        "prompt_lens": prompt_lens
    }