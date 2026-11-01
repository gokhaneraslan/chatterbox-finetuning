import os
import requests
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from src.config import TrainConfig


cfg = TrainConfig()
DEST_DIR = cfg.model_dir

# Optional: the repos are public. A token is only needed if a download fails with 401/403.
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Paralinguistic tags that must remain single tokens after merging
CHECK_TAGS = ["[laugh]", "[chuckle]", "[cough]"]

CHATTERBOX_NANO_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/ve.safetensors?download=true",
    "t3_nano_v1.safetensors": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/t3_nano_v1.safetensors?download=true",
    "s3gen_meanflow.safetensors": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/s3gen_meanflow.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/conds.pt?download=true",
    "t3_nano_v1.yaml": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/t3_nano_v1.yaml?download=true",
    "vocab.json": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/vocab.json?download=true",
    "added_tokens.json": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/added_tokens.json?download=true",
    "special_tokens_map.json": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/special_tokens_map.json?download=true",
    "tokenizer_config.json": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/tokenizer_config.json?download=true",
    "merges.txt": "https://huggingface.co/ResembleAI/chatterbox-nano/resolve/main/merges.txt?download=true",
    "grapheme_mtl_merged_expanded_v1.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}

CHATTERBOX_TURBO_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/ve.safetensors?download=true",
    "t3_turbo_v1.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/t3_turbo_v1.safetensors?download=true",
    "s3gen_meanflow.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen_meanflow.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/conds.pt?download=true",
    "vocab.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/vocab.json?download=true",
    "added_tokens.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/added_tokens.json?download=true",
    "special_tokens_map.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/special_tokens_map.json?download=true",
    "tokenizer_config.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/tokenizer_config.json?download=true",
    "merges.txt": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/merges.txt?download=true",
    "t3_turbo_v1.yaml": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/t3_turbo_v1.yaml?download=true",
    "grapheme_mtl_merged_expanded_v1.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}

CHATTERBOX_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors?download=true",
    "t3_cfg.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt?download=true",
    "tokenizer.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}

def download_file(url, dest_path):
    """Downloads a file with a progress bar. Uses a .part file to avoid keeping partial downloads."""

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading: {os.path.basename(dest_path)}...")
    tmp_path = dest_path + ".part"

    try:
        response = requests.get(url, stream=True, headers=HEADERS)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(tmp_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024 * 1024):
                bar.update(file.write(data))

        os.replace(tmp_path, dest_path)
        print(f"Download complete: {dest_path}\n")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            print("Access denied. Log in with 'huggingface-cli login' or set the HF_TOKEN environment variable.")
        sys.exit(1)


def merge_and_save_turbo_tokenizer():
    """
    Merges the base tokenizer with the multilingual grapheme vocab and
    overwrites the tokenizer files in DEST_DIR.
    Nano: base is the original Nano tokenizer (keeps paralinguistic tag IDs).
    Turbo/Standard: base is plain GPT-2, as in the original kit.
    """
    print("\n--- Vocab Merging Begins ---")

    try:
        base_src = DEST_DIR if cfg.is_nano else "gpt2-medium"
        base_tokenizer = AutoTokenizer.from_pretrained(base_src)
    except Exception as e:
        print(f"ERROR: The base tokenizer could not be loaded -> {e}")
        return 0

    initial_len = len(base_tokenizer)
    print(f"   Original Size: {initial_len}")

    # Nano tags must keep their original IDs (also guards against a tokenizer merged earlier on plain GPT-2)
    nano_tag_ids = {"[laugh]": 50275, "[chuckle]": 50274, "[cough]": 50270}
    if cfg.is_nano:
        wrong = {t: base_tokenizer.convert_tokens_to_ids(t) for t, i in nano_tag_ids.items()
                 if base_tokenizer.convert_tokens_to_ids(t) != i}
        if wrong:
            print(f"ERROR: Base tokenizer is not the original Nano tokenizer: {wrong}")
            print(f"Delete '{DEST_DIR}' and run setup.py again.")
            return 0

    custom_vocab_path = os.path.join(DEST_DIR, "grapheme_mtl_merged_expanded_v1.json")
    print(f"Loading: Custom Vocab ({custom_vocab_path})")

    with open(custom_vocab_path, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)

    if "model" in custom_data and "vocab" in custom_data["model"]:
        vocab_dict = custom_data["model"]["vocab"]
    else:
        print("Warning: The custom vocab format differs from what is expected.")
        return 0

    base_tokenizer.add_tokens(list(vocab_dict.keys()))
    final_len = len(base_tokenizer)

    print(f"Merging: {final_len - initial_len} new tokens added.")
    print(f"   New Dimension: {final_len}")

    print(f"Saving: Writing the merged tokenizer to '{DEST_DIR}'...")
    base_tokenizer.save_pretrained(DEST_DIR)

    # Tags must stay single tokens (Nano: with their original IDs)
    tag_ids = {t: base_tokenizer.convert_tokens_to_ids(t) for t in CHECK_TAGS}
    single = all(len(base_tokenizer.encode(t, add_special_tokens=False)) == 1 for t in CHECK_TAGS)
    ids_ok = (not cfg.is_nano) or all(tag_ids[t] == nano_tag_ids[t] for t in CHECK_TAGS)

    if single and ids_ok:
        print(f"Tags OK: {tag_ids}")
    else:
        print(f"WARNING: Paralinguistic tags changed: {tag_ids}")

    print("MERGE SUCCESSFUL!")

    return final_len


def test_merge_tokenizer_process(tokenizer_path):

    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path)

        print(f"--- RESULTS ---")
        print(f"Folder: {tokenizer_path}")
        print(f"Actual Vocab Size (len): {len(tok)}")

        test_token = "[ta]"
        test_id = tok.encode(test_token, add_special_tokens=False)

        print(f"Test Token '{test_token}' ID: {test_id}")

        if len(tok) > 50276:
            print("SUCCESS! New tokens have been added.")
        else:
            print("ERROR: The size still appears old.")

    except Exception as e:
        print(f"Error: {e}")


def main():

    print("--- Chatterbox Pretrained Model Setup ---\n")

    # 1. Create the directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
    else:
        print(f"Directory found: {DEST_DIR}")

    if cfg.is_turbo:
        if cfg.is_nano:
            print(f"Mode: CHATTERBOX-NANO (Checking {len(CHATTERBOX_NANO_FILES)} files)")
            FILES_TO_DOWNLOAD = CHATTERBOX_NANO_FILES
        else:
            print(f"Mode: CHATTERBOX-TURBO (Checking {len(CHATTERBOX_TURBO_FILES)} files)")
            FILES_TO_DOWNLOAD = CHATTERBOX_TURBO_FILES
    else:
        print(f"Mode: CHATTERBOX-TTS (Checking {len(CHATTERBOX_FILES)} files)")
        FILES_TO_DOWNLOAD = CHATTERBOX_FILES

    # 2. Download files
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = os.path.join(DEST_DIR, filename)
        download_file(url, dest_path)

    # 3. Merge tokenizer (Turbo/Nano only)
    if cfg.is_turbo:
        label = "NANO" if cfg.is_nano else "TURBO"
        new_vocab_size = merge_and_save_turbo_tokenizer()

        if new_vocab_size > 0:
            print("\n" + "=" * 60)
            print(f"INSTALLATION COMPLETE (CHATTERBOX-{label} MODE)")
            print(f"All models are set up in '{DEST_DIR}/' folder.")
            print("Please update the 'new_vocab_size' value in 'src/config.py'")
            print(f"to: {new_vocab_size}")
            print("=" * 60 + "\n")
        else:
            print("ERROR: Tokenizer merge failed. Check the messages above.")
            sys.exit(1)

    else:
        print("\nINSTALLATION COMPLETE (CHATTERBOX-TTS MODE)")
        print(f"All models are set up in '{DEST_DIR}/' folder.")
        print("Note: 'grapheme_mtl_merged_expanded_v1.json' was saved as 'tokenizer.json' for the new vocabulary.")


if __name__ == "__main__":
    main()