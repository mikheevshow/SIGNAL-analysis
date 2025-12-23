from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from tqdm import tqdm
from src.tokenizer_utils import tokenize, load_tokenizer
from transformers import PreTrainedTokenizerFast

@dataclass
class ModelConfig:
    model_name_or_path: str
    instruct: bool

def compute_surprisal(logits, token_ids):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    token_probs = probs[np.arange(len(token_ids)), token_ids]
    surprisal = -np.log2(token_probs)
    return surprisal[0]


def word_level_hidden_state(sentence_hidden_states: np.ndarray,
                            sentence: str,
                            word_position: int,
                            tokenizer: PreTrainedTokenizerFast,
                            strategy: Literal["avg", "max_surprisal"] = "avg",
                            logits: np.ndarray = None,
                            verbose: bool = False) -> tuple[np.ndarray, dict]:

    tokenizer_output = tokenize(
        tokenizer=tokenizer,
        sentences=[sentence],
    )

    words = sentence.split(" ")
    word = words[word_position]
    word_index = sentence.index(word)

    offset_mappings = tokenizer_output["offset_mapping"][0].tolist()
    offset_mapping_indices = []

    for i, offset_mappings in enumerate(offset_mappings):
        if offset_mappings[0] == offset_mappings[1] and offset_mappings[0] == 0:
            if verbose:
                if i == 0:
                    print("Skipp heading service token")
                else:
                    print("Skipp trailing service token")
            continue
        if offset_mappings[0] >= word_index - 1:
            if word_position == len(words) - 1:
                offset_mapping_indices.append(i)
            else:
                next_word = words[word_position + 1]
                next_word_index = sentence.index(next_word)
                if offset_mappings[1] <= next_word_index:
                    offset_mapping_indices.append(i)
                else:
                    break

    word_info = {
        "max_surprisal_relative_index": -1,
        "word_length_tokens": len(offset_mapping_indices),
    }

    if strategy == "avg":
        final_embedding = sentence_hidden_states[offset_mapping_indices, :, :].mean(axis=0)
    elif strategy == "max_surprisal":
        if logits is None:
            raise RuntimeError("Logits cannot be None when strategy=max_surprisal")
        surprisal = compute_surprisal(logits=logits, token_ids=tokenizer_output["input_ids"])
        local_argmax = np.argmax(surprisal[offset_mapping_indices])
        word_info["max_surprisal_relative_index"] = local_argmax
        max_surprisal_index = offset_mapping_indices[local_argmax]
        final_embedding = sentence_hidden_states[max_surprisal_index, :, :]
    else:
        raise ValueError("Strategy not implemented")

    return final_embedding, word_info

if __name__ == "__main__":

    model_list = [
        # ModelConfig(model_name_or_path="RefalMachine/RuadaptQwen3-4B-Instruct", instruct=True),
        # ModelConfig(model_name_or_path="Qwen/Qwen3-4B-Instruct-2507", instruct=True),
        # ModelConfig(model_name_or_path="RefalMachine/RuadaptQwen2.5-14B-Instruct", instruct=True),
        # ModelConfig(model_name_or_path="meta-llama/Meta-Llama-3-8B", instruct=False),
        ModelConfig(model_name_or_path="Qwen/Qwen2.5-7B", instruct=False),
        # ModelConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", instruct=True),
        # ModelConfig(model_name_or_path="RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct", instruct=True),
        # ModelConfig(model_name_or_path="mistralai/Mistral-7B-v0.1", instruct=False),
        # ModelConfig(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1", instruct=True),
        ModelConfig(model_name_or_path="ai-sage/GigaChat3-10B-A1.8B-base", instruct=False),
    ]

    stimuli_path = "../hf_datasets/stimuli.csv"
    stimuli_df = pd.read_csv(stimuli_path)

    for model_cfg in model_list:

        strategy = "max_surprisal"

        tokenizer = load_tokenizer(model_cfg.model_name_or_path)

        results_folder = f"./results/{model_cfg.model_name_or_path.replace('/', '_')}"
        result_csv = results_folder + "/results.csv"

        df = pd.read_csv(result_csv)

        df = df.merge(stimuli_df, how="left", on="sentence")

        words_info = {
            "max_surprisal_relative_index": [],
            "word_length_tokens": [],
            "target": []
        }

        for i, row in tqdm(df.iterrows()):

            hidden_states_path = row["hidden_states_path"]
            hidden_states = np.load(hidden_states_path)

            if strategy == "max_surprisal":
                logits_path = row["logits"]
                logits = np.load(logits_path)
            else:
                logits = None

            print("Hidden state shape: ", hidden_states.shape)

            sentence = row["sentence"]
            word_position = int(row["position"])

            print(sentence)
            print(word_position)

            aggregated_word_level_hidden_states, word_info = word_level_hidden_state(
                sentence_hidden_states=hidden_states,
                sentence=sentence,
                word_position=word_position,
                tokenizer=tokenizer,
                strategy=strategy,
                logits=logits,
            )

            # print(aggregated_word_level_hidden_states.shape)

            # if strategy == "max_surprisal":
            #     word_level_path = hidden_states_path.split(".npy")[0] + "_max_surprisal_hidden_states.npy"
            # else:
            #     word_level_path = hidden_states_path.split(".npy")[0] + "_word_level_hidden_states.npy"
            # np.save(word_level_path, aggregated_word_level_hidden_states)

            words_info["max_surprisal_relative_index"].append(word_info["max_surprisal_relative_index"])
            words_info["word_length_tokens"].append(word_info["word_length_tokens"])
            words_info["target"].append(row["target"])

        pd.DataFrame(words_info).to_csv(f"./tokens_{model_cfg.model_name_or_path.replace('/', '_')}.csv", index=False)
