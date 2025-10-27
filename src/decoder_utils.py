import csv
import logging
import os
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from text_utils import find_index_of_incongruent_word
from tokenizer_utils import load_tokenizer, tokenize

logger = logging.getLogger(__name__)

AggT = Literal["offset", "onset", "mean", "sum"]

@dataclass
class ModelConfig:
    model_name_or_path: str
    instruct: bool

@torch.no_grad()
def _extract_word_hidden_states(
    texts: List[str],
    model_config: ModelConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
) -> list:

    tokenizer = load_tokenizer(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    model.eval().to(device)

    results = []

    for batch_start in range(0, len(texts), batch_size):

        batch = texts[batch_start: batch_start + batch_size]

        encoded_tokens = tokenize(
            tokenizer=tokenizer,
            sentences=batch,
            use_chat_template=model_config.instruct,
        )

        encoded_tokens["input_ids"] = encoded_tokens["input_ids"].to(device)
        encoded_tokens["attention_mask"] = encoded_tokens["attention_mask"].to(device)

        with torch.inference_mode():
            output = model(
                input_ids=encoded_tokens["input_ids"],
                attention_mask=encoded_tokens["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

        # [batch, layers, tokens, dim] -> [batch, tokens, layers, dim]
        hidden_states = torch.stack(output.hidden_states, dim=1).permute(0, 2, 1, 3)

        for text, hs, offsets in zip(batch, hidden_states, encoded_tokens["offset_mapping"]):
            results.append({
                "text": text,
                "hidden_states": hs.cpu().detach(),
                "offset_mapping": offsets,
            })

        logger.info(f"Processed batch {batch_start // batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")

    return results



def calculate_aggregated_hidden_states(
    congruent_sentences: list[str],
    incongruent_sentences: list[str],
    model_name: str = "openai-community/gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    aggregation_method: AggT = "mean",
) -> list:

    indices_of_incongruent_word = find_index_of_incongruent_word(congruent_sentences=congruent_sentences,
                                                                 incongruent_sentences=incongruent_sentences)

    logger.info(indices_of_incongruent_word)

    outputs = _extract_word_hidden_states(texts=incongruent_sentences,
                                          model_name=model_name,
                                          device=device)

    for output, incongruent_word_start_end in zip(outputs, indices_of_incongruent_word):

        output["incongruent_word_start_end_indices"] = incongruent_word_start_end

        offset_mapping = output["offset_mapping"]

        incongruent_words_embedding_indices = []
        for i, om in enumerate(offset_mapping):
            if om[0] >= incongruent_word_start_end[0] and om[1] <= incongruent_word_start_end[1]:
                incongruent_words_embedding_indices.append(i)

        sentence_hidden_states = output["hidden_states"]
        incongruent_word_hidden_states = sentence_hidden_states[incongruent_words_embedding_indices]

        if aggregation_method == "mean":
            output["aggregated_word_embedding_by_layer"] = incongruent_word_hidden_states.mean(dim=0)
        else:
            raise NotImplementedError(f"Aggregation method {aggregation_method} not implemented")

    return outputs



def calculate_hidden_states_all_tokens(
    tokenizer,
    model,
    congruent_sentences: list[str],
    incongruent_sentences: list[str],
    model_config: ModelConfig,
    save_dir: str = "./results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "results.csv")

    processed_texts = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_texts.add(row["sentence"])
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence", "model", "instruct", "hidden_states_path"])

    remaining_sentences = [s for s in incongruent_sentences if s not in processed_texts]
    if not remaining_sentences:
        logger.info("All sentences processed. Skipping all sentences")
        return csv_path

    logger.info(f"Processing {len(remaining_sentences)} new sentences "
                f"for model {model_config.model_name_or_path}")


    for batch_start in range(0, len(remaining_sentences), batch_size):
        batch = remaining_sentences[batch_start: batch_start + batch_size]

        encoded_tokens = tokenize(
            tokenizer=tokenizer,
            sentences=batch,
            use_chat_template=model_config.instruct,
        )

        encoded_tokens["input_ids"] = encoded_tokens["input_ids"].to(device)
        encoded_tokens["attention_mask"] = encoded_tokens["attention_mask"].to(device)

        with torch.inference_mode():
            output = model(
                input_ids=encoded_tokens["input_ids"],
                attention_mask=encoded_tokens["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = torch.stack(output.hidden_states, dim=1).permute(0, 2, 1, 3)

        for idx, (text, hs) in enumerate(zip(batch, hidden_states)):
            sentence_idx = len(processed_texts) + batch_start + idx
            base_name = f"{sentence_idx:05d}_{model_config.model_name_or_path.replace('/', '_')}.npy"
            hs_path = os.path.join(save_dir, base_name)
            np.save(hs_path, hs.cpu().numpy())

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([text, model_config.model_name_or_path, model_config.instruct, hs_path])

        logger.info(f"Batch processed {batch_start // batch_size + 1} "
                    f"of {(len(remaining_sentences) - 1) // batch_size + 1}")

    logger.info(f"Done. All results saved in {save_dir}")
    return csv_path


if __name__ == "__main__":

    aggregate_tokens = False

    model_list = [
        # ModelConfig(model_name_or_path="RefalMachine/RuadaptQwen3-4B-Instruct", instruct=True),
        # ModelConfig(model_name_or_path="Qwen/Qwen3-4B-Instruct-2507", instruct=True),
        # ModelConfig(model_name_or_path="RefalMachine/RuadaptQwen2.5-14B-Instruct", instruct=True),
        # ModelConfig(model_name_or_path="meta-llama/Meta-Llama-3-8B", instruct=False),
        # ModelConfig(model_name_or_path="Qwen/Qwen2.5-7B", instruct=False),
        # ModelConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", instruct=True),
        # ModelConfig(model_name_or_path="RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct", instruct=True),
        # ModelConfig(model_name_or_path="mistralai/Mistral-7B-v0.1", instruct=False),
        # ModelConfig(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1", instruct=True),
    ]

    ds = load_dataset("ContributorsSIGNAL/SIGNAL")
    df = ds["train"].to_pandas()

    logger.info(df.columns)

    congruent_sentences = df["congruent"].tolist()
    incongruent_sentences = df["sentence"].tolist()

    for model_cfg in tqdm(model_list, desc="Model"):

        tokenizer = load_tokenizer(model_cfg.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name_or_path, output_hidden_states=True)
        model.eval().to("cpu")

        calculate_hidden_states_all_tokens(
            tokenizer,
            model,
            congruent_sentences=congruent_sentences,
            incongruent_sentences=incongruent_sentences,
            model_config=model_cfg,
            save_dir=f"./results/{model_cfg.model_name_or_path.replace('/', '_')}",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=600,
        )
