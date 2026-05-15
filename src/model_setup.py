"""
Load the base Qwen model with LoRA adapters attached.

Two backends, selected by the UNSLOTH env var (see config.USE_UNSLOTH):

* unsloth + vLLM: fast inference via `model.fast_generate`, 4-bit via unsloth. Intended path on Vocareum / WSL2 Linux.
* transformers+PEFT: portable fallback. Uses HF `model.generate` and loads adapters via `PeftModel.from_pretrained`. Slower but runs without unsloth/vLLM.

Both backends return (model, tokenizer). Use `generate()` and `load_with_lora()` from this module to stay backend-agnostic.
"""

from config import (
    GPU_MEMORY_UTILIZATION,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    USE_UNSLOTH,
)


def load_base_model():
    """Return (model, tokenizer) with LoRA adapters attached for training."""
    if USE_UNSLOTH:
        return _load_unsloth()
    return _load_transformers()


def _load_unsloth():
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        # WSL local: torch 2.10 + vllm 0.19 hits a torch.compile bug in the CUDA-graph backend. 
        # enforce_eager=True skips graph compilation and generates fine, just slightly slower. 
        # Probably safe to remove on Vocareum.
        enforce_eager=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def _load_transformers():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def generate(model, tokenizer, text: str, max_tokens: int = 1024,
             temperature: float = 0.8, top_p: float = 0.95,
             lora_request=None) -> str:
    """Backend-agnostic text completion.

    `lora_request` is only meaningful on the unsloth backend (obtain via`model.load_lora(path)`). 
    On the transformers backend, load the adapter with `load_adapter()` before calling this.
    """
    if USE_UNSLOTH:
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        return model.fast_generate([text], sampling_params=sampling_params, lora_request=lora_request)[0].outputs[0].text

    import torch

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def load_adapter(model, adapter_dir: str):
    """Attach a saved LoRA adapter for inference. 
    Returns a lora_request handle to pass to `generate()` (unsloth), or None (transformers; adapter is merged into `model` in place and the active adapter is switched).
    """
    if USE_UNSLOTH:
        return model.load_lora(adapter_dir)

    from peft import PeftModel  # noqa: F401 — ensures peft is loaded

    model.load_adapter(adapter_dir, adapter_name="grpo")
    model.set_adapter("grpo")
    return None


def disable_adapter(model) -> None:
    """Switch to base-model-only inference."""
    if USE_UNSLOTH:
        return
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()


def enable_adapter(model) -> None:
    if USE_UNSLOTH:
        return
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()


if __name__ == "__main__":
    model, tokenizer = load_base_model()

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"backend: {'unsloth+vLLM' if USE_UNSLOTH else 'transformers+PEFT'}")
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    print(f"lora_rank={LORA_RANK}  target_modules={LORA_TARGET_MODULES}")
