"""Local text generation using the shared CPU model."""

from functools import lru_cache
from copy import deepcopy
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LLM_MODEL, LLM_MAX_INPUT_TOKENS, LLM_MAX_NEW_TOKENS

tokenizer_lock = Lock()


@lru_cache(maxsize=1)
def load_model():
    """Cache one tokenizer and model per process."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=torch.float32)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_text(messages: list[dict], max_new_tokens: int = LLM_MAX_NEW_TOKENS) -> str:
    """Generate from role/content messages without retaining conversation state."""
    if not messages:
        raise ValueError("Provide at least one message.")
    if not 1 <= max_new_tokens <= LLM_MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {LLM_MAX_NEW_TOKENS}.")

    tokenizer, model = load_model()
    # Serialize shared tokenizer access because its settings can mutate.
    with tokenizer_lock:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    input_length = inputs["input_ids"].shape[1]
    if input_length > LLM_MAX_INPUT_TOKENS:
        raise ValueError("Prompt is too long. Reduce context or history before generating.")
    if input_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt and output exceed the model's context window.")

    generation_config = deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    # Causal models return prompt tokens followed by generated tokens.
    new_tokens = output[0, input_length:]
    with tokenizer_lock:
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
