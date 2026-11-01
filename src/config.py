from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainConfig:
    # --- Model selection ---
    is_turbo: bool = True  # True: Turbo (GPT-2 medium). False: Standard.
    is_nano: bool = True   # True: Nano (GPT-2 small). Requires is_turbo=True.
    
    # --- Training strategy ---
    is_lora: bool = True  # True: LoRA (recommended for < 10h data). False: full fine-tune.
    
    # --- Paths ---
    # Directory where setup.py downloads the base model files
    model_dir: str = "./pretrained_models_nano" if is_nano else "./pretrained_models"
    # Path to your metadata CSV (Format: ID|RawText|NormText)
    #csv_path: str = "./MyTTSDataset/metadata.csv"
    metadata_path: str = "./MyTTSDataset/metadata_final.json"
    # Directory containing WAV files
    wav_dir: str = "./maya1_dataset/wavs"
    #wav_dir: str = "./FileBasedDataset"
    preprocessed_dir = "./MyTTSDataset/preprocess_nano" if is_nano else "./MyTTSDataset/preprocess"
    #preprocessed_dir = "./FileBasedDataset/preprocess"
    # Output directory for the finetuned model
    output_dir: str = "./chatterbox_output_nano" if is_nano else "./chatterbox_output"
    
    is_inference = True
    inference_prompt_path: str = "./speaker_reference/2.wav"
    inference_test_text: str = "Merhaba, sesimi geliştirmem oldukça uzun zaman aldı ve şimdi sahip olduğuma göre, sessiz kalmayacağım."

    # --- Dataset ---
    ljspeech = False      # True: LJSpeech format (metadata.csv)
    json_format = True  # True: JSON format. False (with ljspeech=False): file-based.
    preprocess = True    # Set False if the dataset is already preprocessed.

    lora_r: int = 16
    lora_alpha: int = 32
    turbo_lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj", "c_fc", "spkr_enc"])
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "spkr_enc"])
    lora_modules_to_save: List[str] = field(default_factory=lambda: ["text_emb", "text_head"])

    # --- Vocabulary ---
    # Must equal len(tokenizer). Turbo/Nano: use the value printed by setup.py.
    new_vocab_size: int = 52274 if is_nano else 52260 if is_turbo else 2454  # Turbo/Nano: use the value printed by setup.py

    # --- Hyperparameters ---
    batch_size: int = 32   # Adjust to VRAM
    grad_accum: int = 1    # Effective batch = batch_size * grad_accum
    learning_rate: float = 1e-4 if is_lora else 1e-5
    num_epochs: int = 10 if is_lora else 30
    save_steps: int = 500
    save_total_limit: int = 5
    dataloader_num_workers: int = 4

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850     # Longer audio is truncated
    prompt_duration: float = 3.0  # Reference prompt length (seconds)

    def __post_init__(self):
        assert not (self.is_nano and not self.is_turbo), "is_nano=True requires is_turbo=True"

    @property
    def full_ft_filename(self):
        return ("t3_nano_finetuned.safetensors" if self.is_nano else
                "t3_turbo_finetuned.safetensors" if self.is_turbo else
                "t3_finetuned.safetensors")

    @property
    def merged_filename(self):
        return self.full_ft_filename.replace("finetuned", "finetuned_merged")