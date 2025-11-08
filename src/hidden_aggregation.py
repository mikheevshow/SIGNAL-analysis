import os
import torch


import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from tokenizer_utils import load_tokenizer, tokenize

instruct_dict: dict[str, bool] = {
    "meta-llama/Llama-3.1-8B": False,
    "mistralai/Mistral-7B-v0.1": False,
    "Qwen/Qwen2.5-7B": False,
}

if __name__ == "__main__":

    models_result_folder: list[tuple[str, str]] = [
        # ("meta-llama/Llama-3.1-8B", "./results/meta-llama_Meta-Llama-3-8B"),
        ("mistralai/Mistral-7B-v0.1","./results/mistralai_Mistral-7B-v0.1"),
        # ("Qwen/Qwen2.5-7B", "./results/Qwen_Qwen2.5-7B")
    ]

    stimuli_df = pd.read_csv("../hf_datasets/stimuli.csv")

    for model_and_results in models_result_folder:

        # Ключи номер слоя, значение матрица (n_sentences x emb_dim)
        layers_sentences_embeddings: dict[int, list[np.ndarray]] = dict()

        model_name, hiddens_folder = model_and_results

        # Проверка порядка
        results_df = pd.read_csv(Path(hiddens_folder) / "results.csv")
        for s1, s2 in zip(stimuli_df["sentence"], results_df["sentence"]):
            if s1 != s2:
                raise RuntimeError(f"S1 {s1} != S2 {s2}")

        # Сортировка файлов скрытых состояний
        npy_files: list[tuple[int, str]] = []
        for file in os.listdir(hiddens_folder):
            if file.endswith(".npy"):
                file_index = int(file.split("_")[0])
                npy_files.append((file_index, file))
        npy_files.sort(key=lambda x: x[0])
        npy_files: list[str] = list(map(lambda x: x[1], npy_files))


        tokenizer = load_tokenizer(model_name)

        for sentence, hidden_states_file in tqdm(zip(stimuli_df.sentence, npy_files)):

            # Hidden states dimensins (tokens x layers x emb_dim)
            hiddens = np.load(hiddens_folder + "/" + hidden_states_file)

            tokenization_result = tokenize(
                tokenizer=tokenizer,
                sentences=[sentence],
                use_chat_template=instruct_dict[model_name])

            # Extract last embedding of the sentence

            last_one_index = tokenization_result["attention_mask"][0].numpy().sum() - 1

            embeddings_per_layer = hiddens[last_one_index]

            for layer_idx in range(embeddings_per_layer.shape[0]):
                emb = embeddings_per_layer[layer_idx]
                if layer_idx not in layers_sentences_embeddings:
                    layers_sentences_embeddings[layer_idx] = [emb]
                else:
                    layers_sentences_embeddings[layer_idx].append(emb)

        for k, v in layers_sentences_embeddings.items():
            save_path = Path(hiddens_folder) / "stimuli_matrix_by_layer"
            save_path.mkdir(parents=True, exist_ok=True)
            np.save(save_path / f"{k}.npy", np.vstack(v))
