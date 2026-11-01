import os
import torch
import soundfile as sf
from transformers import TrainerCallback
from safetensors.torch import load_file
from safetensors.torch import save_file

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3
from src.model import resize_and_load_t3_weights
from src.utils import setup_logger, trim_silence_with_vad, load_t3_state


logger = setup_logger("InferenceCallback")


class InferenceCallback(TrainerCallback):

    def __init__(self, config):

        self.config = config
        self.inference_dir = os.path.join(config.output_dir, "inference_samples")
        os.makedirs(self.inference_dir, exist_ok=True)

        if not hasattr(config, 'inference_prompt_path') or not config.inference_prompt_path:
            logger.warning("The inference prompt path is not specified; sampling will be skipped.")
            self.skip_inference = True

        elif not hasattr(config, 'inference_test_text') or not config.inference_test_text:
            logger.warning("The inference test text is not specified; sampling will be skipped.")
            self.skip_inference = True

        else:
            self.skip_inference = False
            logger.info(f"Inference callback is ready. Samples will be saved to: {self.inference_dir}")

    def on_save(self, args, state, control, **kwargs):

        if self.skip_inference:
            return

        step = state.global_step
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        is_lora = getattr(self.config, "is_lora", False)

        if is_lora:
            if not os.path.exists(checkpoint_dir):
                logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
                return

            logger.info(f"Initializing inference for checkpoint-{step} (LoRA)...")

            try:
                logger.info(f"Saving PEFT adapters explicitly to {checkpoint_dir}...")
                model_wrapper = kwargs.get('model')

                peft_model_to_save = None
                if hasattr(model_wrapper, 'model') and isinstance(model_wrapper.model, torch.nn.Module):
                    peft_model_to_save = model_wrapper.model
                elif hasattr(model_wrapper, 't3'):
                    peft_model_to_save = model_wrapper.t3
                else:
                    peft_model_to_save = model_wrapper

                if hasattr(peft_model_to_save, 'save_pretrained'):
                    peft_model_to_save.save_pretrained(checkpoint_dir)
                    logger.info("Adapter config and weights saved successfully.")
                else:
                    logger.warning("No save_pretrained method found on the model.")

            except Exception as e:
                logger.error(f"Failed to force save PEFT adapters: {e}")

            try:
                output_path = os.path.join(self.inference_dir, f"checkpoint-{step}.wav")
                self._generate_sample_lora(checkpoint_dir, output_path)
            except Exception as e:
                logger.error(f"Error during LoRA inference (step {step}): {e}", exc_info=True)

        else:

            weights_path = os.path.join(checkpoint_dir, "model.safetensors")
            if not os.path.exists(weights_path):
                weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

            if not os.path.exists(weights_path):
                logger.warning(f"Checkpoint weights not found: {checkpoint_dir}")
                return

            logger.info(f"Initializing inference for checkpoint-{step} (Full Fine-Tune)...")

            try:
                output_path = os.path.join(self.inference_dir, f"checkpoint-{step}.wav")
                self._generate_sample_full(checkpoint_dir, weights_path, output_path)
            except Exception as e:
                logger.error(f"Error during inference (step {step}): {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # LoRA inference
    # -------------------------------------------------------------------------
    def _generate_sample_lora(self, checkpoint_dir: str, output_path: str):

        from peft import PeftModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_turbo = getattr(self.config, "is_turbo", False)
        EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS

        inference_engine = None
        new_t3 = None

        try:
            # Rebuild the base T3 with the resized vocab
            temp_original = EngineClass.from_local(self.config.model_dir, device="cpu")
            pretrained_state = temp_original.t3.state_dict()
            new_config = temp_original.t3.hp

            new_config.text_tokens_dict_size = self.config.new_vocab_size
            if hasattr(new_config, "use_cache"):
                new_config.use_cache = False

            new_t3 = T3(hp=new_config)
            new_t3 = resize_and_load_t3_weights(new_t3, pretrained_state)

            if is_turbo and hasattr(new_t3.tfmr, "wte"):
                del new_t3.tfmr.wte

            del temp_original
            del pretrained_state

            inference_engine = EngineClass.from_local(self.config.model_dir, device="cpu")
            inference_engine.t3 = new_t3

            logger.info(f"Loading LoRA adapters from: {checkpoint_dir}")
            inference_engine.t3 = PeftModel.from_pretrained(
                inference_engine.t3,
                checkpoint_dir,
                is_trainable=False,
            )

            inference_engine.t3.to(device).eval()
            inference_engine.s3gen.to(device).eval()
            inference_engine.ve.to(device).eval()
            inference_engine.device = device

            params = {"temperature": 0.8, "repetition_penalty": 1.2}
            if not is_turbo:
                params["cfg_weight"] = 0.5
                params["exaggeration"] = 0.5

            with torch.no_grad():
                wav = inference_engine.generate(
                    text=self.config.inference_test_text,
                    audio_prompt_path=self.config.inference_prompt_path,
                    **params,
                )

            if isinstance(wav, tuple):
                wav = wav[0]

            wav_np = wav.squeeze().cpu().numpy()
            trimmed_wav = trim_silence_with_vad(wav_np, inference_engine.sr)
            sf.write(output_path, trimmed_wav, inference_engine.sr)
            logger.info(f"Sample saved: {output_path}")

        except Exception as e:
            logger.error(f"LoRA inference callback failed: {e}", exc_info=True)

        finally:
            if inference_engine:
                del inference_engine
            if new_t3:
                del new_t3
            torch.cuda.empty_cache()
            logger.info("LoRA inference cleanup done.")

    # -------------------------------------------------------------------------
    # Full fine-tune inference
    # -------------------------------------------------------------------------
    def _generate_sample_full(self, checkpoint_dir: str, weights_path: str, output_path: str):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_turbo = getattr(self.config, "is_turbo", False)
        EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS

        tts_engine = EngineClass.from_local(self.config.model_dir, device="cpu")

        t3_config = tts_engine.t3.hp
        if hasattr(self.config, 'new_vocab_size'):
            t3_config.text_tokens_dict_size = self.config.new_vocab_size

        new_t3 = T3(hp=t3_config)

        if is_turbo and hasattr(new_t3.tfmr, "wte"):
            del new_t3.tfmr.wte

        if weights_path.endswith(".safetensors"):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        # Raises if any weight is missing (the sample would be noise)
        load_t3_state(new_t3, state_dict)
        logger.info("All weights loaded successfully.")

        tts_engine.t3 = new_t3

        model_path = os.path.join(checkpoint_dir, self.config.full_ft_filename)
        save_file(tts_engine.t3.state_dict(), model_path)
        logger.info(f"Full T3 model saved to: {model_path}")

        tts_engine.t3.to(device).eval()
        tts_engine.s3gen.to(device).eval()
        tts_engine.ve.to(device).eval()
        tts_engine.device = device

        params = {"temperature": 0.8, "repetition_penalty": 1.2}
        if not is_turbo:
            params["cfg_weight"] = 0.2
            params["exaggeration"] = 1.2

        with torch.no_grad():
            wav = tts_engine.generate(
                text=self.config.inference_test_text,
                audio_prompt_path=self.config.inference_prompt_path,
                **params,
            )

        if isinstance(wav, tuple):
            wav = wav[0]

        wav_np = wav.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, tts_engine.sr)
        sf.write(output_path, trimmed_wav, tts_engine.sr)
        logger.info(f"Sample saved: {output_path}")

        del tts_engine
        del new_t3
        del state_dict
        torch.cuda.empty_cache()