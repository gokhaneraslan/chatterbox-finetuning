import os
from safetensors.torch import save_file
from peft import PeftModel

from src.config import TrainConfig
from src.model import resize_and_load_t3_weights
from src.utils import setup_logger

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3

logger = setup_logger("ChatterboxMerge")


def main():

    cfg = TrainConfig()
    logger.info("--- Starting LoRA Merge Process ---")

    adapter_dir = os.path.join(cfg.output_dir, "new_lang_adapter")
    output_file = os.path.join(cfg.output_dir, cfg.merged_filename)

    if not os.path.exists(adapter_dir):
        logger.error(f"Adapter directory not found at {adapter_dir}! Please train the model first.")
        return

    logger.info("1. Loading base model...")
    EngineClass = ChatterboxTurboTTS if cfg.is_turbo else ChatterboxTTS
    tts_engine_base = EngineClass.from_local(cfg.model_dir, device="cpu")

    if cfg.is_turbo:
        expected = "Nano" if cfg.is_nano else "Turbo"
        if tts_engine_base.model_label != expected:
            raise RuntimeError(f"Config expects {expected}, but '{cfg.model_dir}' contains {tts_engine_base.model_label}.")
        tok_len = len(tts_engine_base.tokenizer)
        if tok_len != cfg.new_vocab_size:
            raise ValueError(f"Tokenizer len {tok_len} != new_vocab_size {cfg.new_vocab_size}. Check the output of setup.py.")

    pretrained_state_dict = tts_engine_base.t3.state_dict()
    t3_config = tts_engine_base.t3.hp

    logger.info(f"2. Preparing T3 model with new vocab size: {cfg.new_vocab_size}...")
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
    new_t3_model = T3(hp=t3_config)
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_state_dict)

    if cfg.is_turbo and hasattr(new_t3_model.tfmr, "wte"):
        logger.info("Deleting tfmr.wte to match training architecture...")
        del new_t3_model.tfmr.wte

    expected_keys = set(new_t3_model.state_dict().keys())

    del tts_engine_base
    del pretrained_state_dict

    logger.info(f"3. Loading LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(new_t3_model, adapter_dir)

    logger.info("4. Merging LoRA weights into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    # The merged model must have exactly the base T3 keys (no LoRA / wrapper leftovers)
    merged_keys = set(merged_model.state_dict().keys())
    if merged_keys != expected_keys:
        raise RuntimeError(
            f"Merged keys differ from base keys. "
            f"Extra: {sorted(merged_keys - expected_keys)[:5]} | Missing: {sorted(expected_keys - merged_keys)[:5]}"
        )

    logger.info(f"5. Saving standalone merged model to {output_file}...")
    save_file(merged_model.state_dict(), output_file)
    logger.info("--- MERGE COMPLETE ---")


if __name__ == "__main__":
    main()