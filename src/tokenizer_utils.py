import logging
import logging_cofig

from typing import List
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding

logger = logging.getLogger(__name__)

def load_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "add_prefix_space"):
        logger.info("Tokenizer adds prefix space")
        tokenizer.add_prefix_space = True
    return tokenizer


def tokenize(
        tokenizer: PreTrainedTokenizerFast,
        sentences: List[str],
        max_length: int = 128,
        use_chat_template: bool = False,
        add_generation_prompt: bool = False,
) -> BatchEncoding:

    if use_chat_template:

        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer does not support chat template, but use_chat_template=True."
            )

        messages_list = [[{"role": "user", "content": s}] for s in sentences]

        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            for messages in messages_list
        ]

        print(texts)

        return tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

    else:
        texts = sentences

        return tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )


if __name__ == "__main__":

    tokenizer = load_tokenizer("mistralai/Mistral-7B-v0.1")

    print(tokenize(
        tokenizer,
        sentences=["Как дела"],
        use_chat_template=False,
    ))