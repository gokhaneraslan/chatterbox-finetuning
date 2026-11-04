# 🎙️ Chatterbox Fine-Tuning & LoRA Toolkit (Nano, Turbo & Standard)

[![GitHub Stars](https://img.shields.io/github/stars/gokhaneraslan/chatterbox-finetuning?style=social)](https://github.com/gokhaneraslan/chatterbox-finetuning)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange.svg)](https://pytorch.org/)

A modular fine-tuning and LoRA toolkit for the **Resemble AI Chatterbox** model family:

- **Chatterbox-Nano** 🚀 *(smallest model, GPT-2 small backbone, built for fast low-memory inference)*
- **Chatterbox-Turbo** ⚡ *(GPT-2 medium backbone)*
- **Chatterbox Standard** 🌐 *(Llama backbone)*

Train a voice or a new language with **LoRA (Low-Rank Adaptation)** or a **Full Fine-Tune**. The kit can keep the original tokenizer or extend it with a multilingual grapheme set, and it renders audio samples while training so you can listen to every checkpoint.

---

## ⚡ Models and Hardware

| Model | Parameters (model card) | Backbone | Tokenizer |
|---|---|---|---|
| **Nano** 🚀 | 110M | GPT-2 small | GPT-2 BPE + tags (50,276) |
| **Turbo** ⚡ | 350M | GPT-2 medium | GPT-2 BPE + tags (50,276) |
| **Standard** 🌐 | 500M | Llama | Grapheme (2,454) |

Notes:
- The parameter column quotes the model cards. This kit measures about **179M** parameters for the Nano T3 (including its embedding tables), so use the measured number when you plan memory.
- Nano and Turbo share the tokenizer and the speech decoder file. Nano only differs in the smaller T3 backbone and its own T3 checkpoint.
- Resemble AI's model card describes Nano as running about 3x faster than realtime on an 8-core CPU.
- The Standard model downloaded by `setup.py` is the original `t3_cfg` checkpoint; the kit swaps in a 23-language grapheme tokenizer for fine-tuning new languages.

**VRAM:** Memory usage is driven by audio sequence length. The Nano model was trained on a 16 GB VRAM GPU with `batch_size = 8` and `grad_accum = 2`. For the Turbo and Standard models, a 24 GB VRAM GPU is recommended using a similar batch configuration.

---

## 💡 Key Features

1. **One toolkit for three models.** Switch between Nano, Turbo and Standard with two flags.
2. **Selective vocabulary extension (`extend_vocab`).** Keep the original tokenizer or merge a multilingual grapheme set into it.
3. **Paralinguistic tags.** Nano and Turbo tokenizers contain tags such as `[laugh]`, `[chuckle]` and `[cough]`. Their token IDs are preserved in both vocabulary modes.
4. **Three dataset formats.** JSON, LJSpeech CSV and file-based (`.wav` + `.txt`).
5. **Separate folders per variant.** Model, preprocessed data and output folders get a suffix derived from your flags, so experiments do not overwrite each other.
6. **Built-in safety checks.** Mismatched model folders, tokenizer sizes and preprocessed data stop with a clear error instead of silently training on bad data.
7. **Live audio monitoring.** An inference callback renders a `.wav` at every checkpoint, trimmed with **Silero VAD**.

---

## 🤖 Choosing the Model

Set the model in `src/config.py`:

| Model | `is_turbo` | `is_nano` |
|---|:---:|:---:|
| Nano | `True` | `True` |
| Turbo | `True` | `False` |
| Standard | `False` | `False` |

`is_nano = True` requires `is_turbo = True`.

Nano is an English base model. Adapting it to another language (such as Turkish) has been tested and works well, but results depend on the language and data quality. We recommend a short pilot run on a subset before a full training.

---

## 🔤 Vocabulary Modes (`extend_vocab`)

For Turbo and Nano, `extend_vocab` decides what `setup.py` does with the tokenizer. It has no effect in Standard mode.

| Setting | What `setup.py` does | `new_vocab_size` |
|---|---|---|
| `extend_vocab = False` | Keeps the original tokenizer. No tokens are added. | 50276 |
| `extend_vocab = True` | Merges the multilingual grapheme set into the original tokenizer. | 52274 |

`new_vocab_size` must equal `len(tokenizer)`. `setup.py` prints the exact value at the end, and the training scripts stop with an error if config and tokenizer disagree.

**What the merge changes (observed on Nano):**
- The merge starts from the model's own tokenizer, so the tag tokens keep their IDs. `setup.py` verifies this and prints `Tags OK`.
- Text is split into smaller pieces. On one Turkish dataset the mean text length grew from 94 to 147 tokens, and English words are split too (for example `Hello` becomes `H`, `el`, `lo`).
- 1,998 new embedding rows are created, initialized with the mean of the existing rows.


**Recommendation:** start with `extend_vocab = False`. Use `True` only if your language has characters or text behavior the original tokenizer handles poorly. If you can, train both on the same subset and compare listening samples; loss values are not comparable across modes because the token counts differ.

**Tags after fine-tuning:** a typical TTS dataset contains no tags, so fine-tuning can weaken them. Test with an English prompt such as `Hi [laugh] there.` after training.

---

## 🗂️ Variants and Folders

The model, preprocessed data, and output folders automatically receive an isolated suffix based on your active configuration flags:

| Variant | Flags | Suffix | `new_vocab_size` | Folder Example |
|---|---|---|---|---|
| **Standard** | `is_turbo=False` | *(none)* | 2454 | `pretrained_models/` |
| **Turbo (Extended)** | `is_turbo=True, is_nano=False, extend_vocab=True` | `_turbo` | 52274 | `pretrained_models_turbo/` |
| **Turbo (Original BPE)** | `is_turbo=True, is_nano=False, extend_vocab=False` | `_turbo_bpe` | 50276 | `pretrained_models_turbo_bpe/` |
| **Nano (Extended)** | `is_turbo=True, is_nano=True, extend_vocab=True` | `_nano` | 52274 | `pretrained_models_nano/` |
| **Nano (Original BPE)** | `is_turbo=True, is_nano=True, extend_vocab=False` | `_nano_bpe` | 50276 | `pretrained_models_nano_bpe/` |

Because every variant maintains its own isolated namespace, **different configurations will never overwrite or collide with each other**. You can download models and run experiments across multiple variants side by side without needing to delete existing folders.

> 💡 **Note on :** setup.py aborts if it finds a merged tokenizer in a folder where extend_vocab = False (for example after copying folders by hand). Delete that folder and run it again.

---

## 📁 Repository Structure

```text
├── pretrained_models*/          # Base weights and tokenizer (one folder per variant)
├── preprocess*/                 # Precomputed tokens and speaker embeddings (preprocessed_dir)
├── chatterbox_output*/          # Adapters, checkpoints and inference samples (output_dir)
├── speaker_reference/           # Reference audio for cloning and the callback
├── MyTTSDataset/                # Example dataset folder (metadata + wavs)
├── src/
│   ├── chatterbox_/             # Vendored Chatterbox model code (Standard, Turbo, Nano)
│   ├── config.py                # All settings and hyperparameters
│   ├── dataset.py               # Dataset loading and collators
│   ├── model.py                 # Vocab-resizing weight transfer and training wrapper
│   ├── inference_callback.py    # Audio sample at every checkpoint
│   ├── preprocess_*.py          # Preprocessors (LJSpeech, JSON, file-based)
│   └── utils.py                 # Logger, Silero VAD, model-file and tokenizer checks
├── train.py                     # Training (HuggingFace Trainer + PEFT)
├── inference.py                 # Speech synthesis
├── merge_lora.py                # Bakes LoRA weights into a standalone file
├── setup.py                     # Downloads models and prepares the tokenizer
└── requirements.txt
```

---

## 🛠️ Installation

### 1. FFmpeg (required)
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey / Scoop)
choco install ffmpeg
scoop install ffmpeg
```

### 2. Python environment
Python 3.10 or newer (3.11 is what upstream Chatterbox is developed on). Check `requirements.txt` for the supported PyTorch versions.
```bash
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning
pip install -r requirements.txt
```

---

## ⚙️ Configuration and Model Download

All settings live in **`src/config.py`**.

**Step 1: choose your variant.**
```python
is_turbo: bool = True        # False: Standard
is_nano: bool = True         # True: Nano (requires is_turbo=True)
extend_vocab: bool = False   # Turbo/Nano only (see "Vocabulary Modes")
```

**Step 2: download the models and prepare the tokenizer.**
```bash
python setup.py
```
The Hugging Face repos are public. If a download fails with 401/403, log in with `huggingface-cli login` or set the `HF_TOKEN` environment variable.

**Step 3: copy the vocabulary size.** `setup.py` ends with a message like:
```text
Please update the 'new_vocab_size' value in 'src/config.py'
to: 50276
```
Set `new_vocab_size` to that number. **Do not skip this step.**
```python
new_vocab_size: int = (52274 if extend_vocab else 50276) if is_turbo else 2454
```

---

## 📊 Dataset Preparation

### Option A: TTS Dataset Generator (recommended)
Use the [TTS Dataset Generator](https://github.com/gokhaneraslan/tts-dataset-generator) to build a dataset from audio or video files:
```bash
git clone https://github.com/gokhaneraslan/tts-dataset-generator.git
cd tts-dataset-generator
pip install -r requirements.txt

python main.py --file your_audio.mp4 --model large --language en --ljspeech True
```
It segments the audio, transcribes it with Whisper and formats the dataset.

### Option B: Your own dataset
Select the format with the flags in `src/config.py`. If several are set, the order of precedence is `ljspeech`, then `json_format`, then file-based.

**JSON** (`ljspeech = False`, `json_format = True`):
```python
metadata_path = "./MyTTSDataset/metadata_final.json"
wav_dir = "./MyTTSDataset/wavs"
```
Every item needs an `id` and a `text`. The audio file must be `wav_dir/{id}.wav`; other fields are ignored.
```json
[
  {"id": "sample_0001", "text": "Merhaba, bu bir Chatterbox Nano testidir."}
]
```

**LJSpeech** (`ljspeech = True`): a pipe-separated CSV at `csv_path` (`filename|raw_text|norm_text`), audio in `wav_dir`.
```text
sample_0001|Hello, this is a test.|Hello, this is a test.
```

**File-based** (`ljspeech = False`, `json_format = False`): `wav_dir` holds `ID.wav` and `ID.txt` pairs.

**Quality guidelines:**
- WAV in any sample rate, mono or stereo (converted to 16 kHz mono automatically)
- 3 to 15 seconds per clip works well; **never above about 34 seconds** (see "Length limits")
- Clean audio, and text that matches what is actually said
- 30+ minutes for a basic voice, several hours for a new language

---

## 🧠 Training

### 1. Strategy: LoRA or Full Fine-Tune

- **`is_lora = True` (recommended, especially below 10 hours of data).** Freezes the base model and trains small adapter matrices plus the text embedding tables (`lora_modules_to_save`). It reduces forgetting and VRAM use. The adapter is saved to `<output_dir>/new_lang_adapter/`. On Nano and Turbo the two embedding tables are trained in full and make up most of the trainable parameters, so the adapter folder also contains the resized embeddings.
- **`is_lora = False`.** Updates every T3 weight. Use it for large datasets. The result is saved as `t3_nano_finetuned.safetensors` (`t3_turbo_finetuned.safetensors` / `t3_finetuned.safetensors` for the other models).

> If a full fine-tune of Turbo or Nano produces static noise or hallucinations in the callback samples, switch to LoRA. `lora_r` is worth tuning; 32 to 64 is a reasonable starting range for Nano.

### 2. Key settings
```python
# --- Training ---
is_lora: bool = True
lora_r: int = 32
lora_alpha: int = 64
turbo_lora_target_modules = ["c_attn", "c_proj", "c_fc", "spkr_enc"]    # Turbo and Nano
lora_modules_to_save = ["text_emb", "text_head"]

# --- Hardware and batch (the Nano reference run: 16 GB GPU) ---
batch_size: int = 8
grad_accum: int = 2          # effective batch = batch_size * grad_accum
num_epochs: int = 10 if is_lora else 20
save_steps: int = 500       # a checkpoint (and sample wav) every 500 steps

# --- Monitoring ---
is_inference = True          # render a sample wav at every checkpoint
inference_prompt_path = "./speaker_reference/2.wav"
inference_test_text = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı."

# --- Dataset ---
preprocess = True            # keep True until this variant's preprocessing has finished once
```

**Choosing `save_steps`:** a checkpoint (and, with `is_inference = True`, a sample wav) is created every `save_steps` steps. For one per epoch use `ceil(num_samples / (batch_size * grad_accum))`. Every checkpoint also stores optimizer state, so watch your disk space (`save_total_limit` keeps only the latest ones).

**Reference audio:** point `inference_prompt_path` to a clean recording in your target language, **longer than 5 seconds** (shorter clips fail an assertion). A native-speaker prompt noticeably helps the samples.

### 3. Launch
```bash
python train.py
```

**What a healthy start looks like in the log:**
- `Mode: CHATTERBOX-NANO` (or TURBO / TTS)
- `Base T3 params` and `New T3 text_emb: (<new_vocab_size>, 768)` for Nano
- `trainable params: ...` (mostly the two embedding tables on Turbo and Nano)
- `Dataset loaded. Total samples: ...` matching your dataset
- no NaN loss, and a `Sample saved: .../checkpoint-N.wav` line at every checkpoint

### 🎧 Judge by listening
The training loss is the sum of the text and speech cross-entropy of the T3 model (the speech decoder is frozen). Its absolute value depends on your data and tokenizer, so do not decide convergence from the number alone. Listen to the callback samples in:
```text
chatterbox_output*/inference_samples/checkpoint-<step>.wav
```

### Built-in safety checks
The scripts stop with a clear error when:
- the model folder does not match your flags (for example Turbo files while `is_nano = True`),
- `len(tokenizer)` differs from `new_vocab_size`,
- preprocessed data was built with a different tokenizer (checked through `preprocess_meta.json`),
- a checkpoint is missing T3 weights, or a merged model does not have exactly the base model's keys.

---

## ⚠️ Important Details

**Preprocessing is mandatory.** The kit extracts speaker embeddings and acoustic tokens, tokenizes the text and stores one `.pt` file per clip, so training only loads tensors. Preprocessed data belongs to one vocabulary mode; each variant has its own folder.

**Length limits.**
- `max_speech_len = 850` speech tokens. At 25 tokens per second this is about **34 seconds**. Longer clips are truncated and **lose their end-of-speech token**, so the model cannot learn to finish those utterances. Filter them out of your metadata.
- `max_text_len = 256` tokens (254 after the start and stop tokens). Longer texts are truncated.

**Sample rates.** Training works at 16 kHz (other rates are resampled automatically); the output audio is 24 kHz.

---

## 🗣️ Inference & Voice Cloning

`inference.py` reads `src/config.py`, rebuilds the model with the right vocabulary and loads the LoRA adapter or the full fine-tuned weights.

Edit the settings at the top of the script:
```python
TEXT_TO_SAY = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."
AUDIO_PROMPT = "./speaker_reference/2.wav"   # clean reference, longer than 5 seconds
```
```bash
python inference.py
```
The result is `output.wav` (24 kHz). The script splits the text into sentences, joins them with short pauses and trims trailing silence and artifacts with Silero VAD. Turbo and Nano ignore `cfg_weight` and `exaggeration`; tune `temperature` and `repetition_penalty` instead.

---

## 📦 Exporting and Merging LoRA Weights

To get one standalone file that needs neither PEFT nor the adapter folder:
```bash
python merge_lora.py
```
This bakes the adapter into the base T3 and writes, inside your `output_dir`:
```text
t3_nano_finetuned_merged.safetensors
# t3_turbo_finetuned_merged.safetensors / t3_finetuned_merged.safetensors for the other models
```
The script verifies that the merged file has exactly the base model's keys. The merged file contains only the T3 weights; the voice encoder, speech decoder and tokenizer still come from the variant's model folder.

---

## 💡 Troubleshooting

| Issue | Cause and solution |
|---|---|
| `Tokenizer len X != new_vocab_size Y`, or `size mismatch` when loading T3 | `new_vocab_size` does not match the tokenizer in the model folder. Use the value printed by `setup.py` and make sure `extend_vocab` matches the folder you ran setup in. |
| `Config expects Nano, but '<folder>' contains Turbo` (or the reverse) | The model folder does not match `is_nano`. Run `python setup.py` with the flags you want. |
| `Data in '<folder>' was built with tokenizer len X, expected Y` | The preprocessed data belongs to another vocabulary mode. Delete that preprocessed folder and run again with `preprocess = True`. |
| `Missing T3 keys` | The checkpoint does not match the rebuilt model (wrong variant or wrong `new_vocab_size`). Check the flags used for that training run. |
| `Expected the original tokenizer (50276 tokens) but found ...` | `extend_vocab = False`, but an earlier run merged the tokenizer in this folder. Delete the folder and run `setup.py` again. |
| `AssertionError: is_nano=True requires is_turbo=True` | Nano shares the Turbo architecture. Set both flags to `True`. |
| `Audio prompt must be longer than 5 seconds!` | Use a longer reference clip. |
| `FileNotFoundError: ... ve.safetensors` | Models are not downloaded yet. Run `python setup.py`. |
| Download fails with 401/403 | Run `huggingface-cli login` or set `HF_TOKEN`. |
| `CUDA out of memory` | Use LoRA, lower `batch_size` and raise `grad_accum`. Gradient checkpointing is already on. The inference callback loads a second copy of the model at every checkpoint, so peak VRAM happens there. |
| bf16 not supported | Use an Ampere-or-newer GPU. |
| No checkpoints or samples appear | `save_steps` is larger than the total step count, or `is_inference` is `False`. |
| The process does not exit after training (seen on macOS) | The adapter or model is already saved; check the output folder. Setting `dataloader_num_workers = 0` avoids DataLoader worker problems. |
| Poor quality or cut-off endings | Check the reference audio, make sure the text matches the audio, and remove clips longer than about 34 seconds. Prefer LoRA for Turbo and Nano. Listen to the samples at every epoch to catch problems early. |

---

## 📜 Acknowledgments and Citation

- Based on [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox)
- Pretrained base models:
  - [ResembleAI / chatterbox (Standard)](https://huggingface.co/ResembleAI/chatterbox)
  - [ResembleAI / chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)
  - [ResembleAI / chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano)

```bibtex
@software{chatterbox_finetuning,
  author = {Gökhan Eraslan},
  title = {Chatterbox Fine-Tuning & LoRA Toolkit (Nano, Turbo & Standard)},
  year = {2025},
  url = {https://github.com/gokhaneraslan/chatterbox-finetuning}
}
```

---

## 📄 License

- This toolkit and fine-tuning scripts are licensed under the [Apache-2.0 License](LICENSE).
- The underlying **Chatterbox TTS** model architecture and code (vendored inside `src/chatterbox_/`) are developed by **Resemble AI** and licensed under the **MIT License**. Original license notices are retained within their respective files.