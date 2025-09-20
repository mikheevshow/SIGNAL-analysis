import logging
import logging_cofig

from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding

logger = logging.getLogger(__name__)

def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "add_prefix_space"):
        logger.info("Tokenizer adds prefix space")
        tokenizer.add_prefix_space = True
    return tokenizer


def tokenize(tokenizer: PreTrainedTokenizerFast,
             sentences: list[str],
             max_length: int = 128) -> BatchEncoding:

    return tokenizer(
        sentences,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        padding="longest",
        # padding_side="left",
        truncation=True,
        max_length=max_length,
    )
