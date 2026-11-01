import os
import sys
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator_turbo, data_collator_standart
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.preprocess_ljspeech import preprocess_dataset_ljspeech
from src.preprocess_file_based import preprocess_dataset_file_based
from src.preprocess_json import preprocess_dataset_json_based
from src.utils import setup_logger, check_pretrained_models
from src.inference_callback import InferenceCallback

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logger("ChatterboxFinetune")


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def main():

    cfg = TrainConfig()

    mode_name = "CHATTERBOX-NANO" if cfg.is_nano else "CHATTERBOX-TURBO" if cfg.is_turbo else "CHATTERBOX-TTS"
    logger.info("--- Starting Chatterbox Finetuning ---")
    logger.info(f"Mode: {mode_name}")
    logger.info(f"Training Strategy: {'LoRA' if cfg.is_lora else 'Full Fine-Tune'}")

    # 0. CHECK MODEL FILES
    mode_check = "chatterbox_nano" if cfg.is_nano else "chatterbox_turbo" if cfg.is_turbo else "chatterbox"
    if not check_pretrained_models(model_dir=cfg.model_dir, mode=mode_check):
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SELECT THE CORRECT ENGINE CLASS
    EngineClass = ChatterboxTurboTTS if cfg.is_turbo else ChatterboxTTS

    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 2. LOAD ORIGINAL MODEL TEMPORARILY (CPU first to save VRAM)
    logger.info("Loading original model to extract weights...")
    tts_engine_original = EngineClass.from_local(cfg.model_dir, device="cpu")

    if cfg.is_turbo:
        # The folder content must match the config flags
        expected = "Nano" if cfg.is_nano else "Turbo"
        found = tts_engine_original.model_label
        if found != expected:
            raise RuntimeError(f"Config expects {expected}, but '{cfg.model_dir}' contains {found}.")

        tok_len = len(tts_engine_original.tokenizer)
        if tok_len != cfg.new_vocab_size:
            raise ValueError(f"Tokenizer len {tok_len} != new_vocab_size {cfg.new_vocab_size}. Check the output of setup.py.")

    logger.info(f"Base T3 params: {count_params(tts_engine_original.t3) / 1e6:.1f}M")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 3. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")

    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    # Prevent caching during training
    if hasattr(new_t3_config, "use_cache"):
        new_t3_config.use_cache = False
    else:
        setattr(new_t3_config, "use_cache", False)

    new_t3_model = T3(hp=new_t3_config)

    # 4. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

    if cfg.is_turbo:
        logger.info(f"{'Nano' if cfg.is_nano else 'Turbo'} Mode: Removing backbone WTE layer...")
        if hasattr(new_t3_model.tfmr, "wte"):
            del new_t3_model.tfmr.wte

    logger.info(f"New T3 text_emb: {tuple(new_t3_model.text_emb.weight.shape)}")

    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 5. PREPARE ENGINE FOR TRAINING
    tts_engine_new = EngineClass.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model

    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters():
        param.requires_grad = False

    for param in tts_engine_new.s3gen.parameters():
        param.requires_grad = False

    # 6. APPLY LORA OR FULL FINE-TUNE
    if cfg.is_lora:
        # --- LoRA PATH ---
        logger.info("Applying LoRA configuration...")

        for param in tts_engine_new.t3.parameters():
            param.requires_grad = False

        from peft import LoraConfig, get_peft_model

        target_modules = cfg.turbo_lora_target_modules if cfg.is_turbo else cfg.lora_target_modules
        logger.info(f"LoRA Target Modules: {target_modules}")
        logger.info(f"Modules to Full Train (Embeddings): {cfg.lora_modules_to_save}")

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=cfg.lora_modules_to_save,
        )

        tts_engine_new.t3 = get_peft_model(tts_engine_new.t3, peft_config)
        tts_engine_new.t3.print_trainable_parameters()

    else:
        # --- FULL FINE-TUNE PATH ---
        logger.info("Full fine-tune: enabling all T3 parameters...")
        tts_engine_new.t3.train()
        for param in tts_engine_new.t3.parameters():
            param.requires_grad = True
        logger.info(f"Trainable params: {count_params(tts_engine_new.t3, trainable_only=True) / 1e6:.1f}M")

    # 7. PREPROCESSING
    if cfg.preprocess:
        logger.info("Initializing Preprocess dataset...")

        if cfg.ljspeech:
            preprocess_dataset_ljspeech(cfg, tts_engine_new)
        elif cfg.json_format:
            preprocess_dataset_json_based(cfg, tts_engine_new)
        else:
            preprocess_dataset_file_based(cfg, tts_engine_new)
    else:
        logger.info("Skipping the preprocessing dataset step...")

    # 8. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    train_ds = ChatterboxDataset(cfg)

    trainer_callbacks = []
    if cfg.is_inference:
        trainer_callbacks.append(InferenceCallback(cfg))

    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    if cfg.is_turbo:
        logger.info("Using Turbo/Nano Data Collator (with dynamic prompt masking)")
        selected_collator = data_collator_turbo
    else:
        logger.info("Using Standard Data Collator")
        selected_collator = data_collator_standart

    # 9. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        remove_unused_columns=False,  # Required for our custom wrapper
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=["tensorboard"],
        fp16=False,
        bf16=True,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=True,  # Reduces VRAM usage by ~60%
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=selected_collator,
        callbacks=trainer_callbacks,
    )

    logger.info("Starting Training Loop...")
    trainer.train()

    # 10. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.is_lora:
        # Save LoRA adapter + resized embeddings via PEFT
        save_path = os.path.join(cfg.output_dir, "new_lang_adapter")
        tts_engine_new.t3.save_pretrained(save_path)
        logger.info(f"LoRA adapter saved to: {save_path}")
        logger.info("NOTE: This adapter contains both LoRA weights AND the new resized embeddings.")
    else:
        # Save full model weights as safetensors
        final_model_path = os.path.join(cfg.output_dir, cfg.full_ft_filename)
        save_file(tts_engine_new.t3.state_dict(), final_model_path)
        logger.info(f"Full model saved to: {final_model_path}")


if __name__ == "__main__":
    main()