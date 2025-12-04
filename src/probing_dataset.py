import os
import pathlib
import pandas as pd
import numpy as np

from pathlib import Path
from src.tokenizer_utils import load_tokenizer, tokenize

# Index(['sentence', 'model', 'instruct', 'hidden_states_path', 'logits',
#        'sentence_id', 'congruent', 'structure', 'length', 'target', 'position',
#        'most_popular', 'percent', 'semantics_grammar', 'semantics', 'grammar',
#        'no', 'unknown', 'subject', 'verb', 'object', 'gen', 'adj',
#        'subject_lemma', 'subject_length', 'subject_gender', 'subject_ipm',
#        'verb_lemma', 'verb_length', 'verb_ipm', 'object_lemma',
#        'object_length', 'object_gender', 'object_ipm', 'gen_lemma',
#        'gen_length', 'gen_gender', 'gen_ipm', 'adj_lemma', 'adj_length',
#        'adj_gender', 'adj_ipm'],
#       dtype='object')

# Функция создания датасета для пробинга на sentence-level уровне
# Необходимо указывать название модели, чтобы вытащить последний токен предложения (без <pad> токенов)
# на основе attention-map
def make_sentence_level_probing_dataset(
        model_name: str,
        model_hiddens_path: Path,
        stimuli_csv_path: Path,
        hiddens_results_csv_output_name: str = "results.csv",
        instruct: bool=False) -> pd.DataFrame:

    hiddens_results_path = model_hiddens_path / hiddens_results_csv_output_name

    results_df = pd.read_csv(hiddens_results_path)
    stimuli_df = pd.read_csv(stimuli_csv_path)

    merged_df = pd.merge(results_df, stimuli_df, how="inner", on="sentence")

    assert merged_df.shape[0] == stimuli_df.shape[0]

    columns_to_select = [
        "sentence_id", "sentence", "congruent", "structure", "length", "position", "target"
    ]

    merged_df = merged_df[columns_to_select + ["hidden_states_path"]]

    result = {col: [] for col in columns_to_select}

    result = {**result, **{"layer": [], "hidden_state": []}}
    tokenizer = load_tokenizer(model_name)

    tokenized_sentences = tokenize(
        tokenizer=tokenizer,
        sentences=merged_df["sentence"].tolist(),
        use_chat_template=instruct,
    )

    for i, row in merged_df.iterrows():

        hidden_states_path = row["hidden_states_path"]
        hidden_states = np.load(os.path.abspath(os.path.join(str(model_hiddens_path.parent.parent), hidden_states_path))) # hidden shape [tokens, layers, emb_dim]

        attention_mask = tokenized_sentences["attention_mask"][i].tolist()
        try:
            first_zero = attention_mask.index(0)
            last_non_pad_idx = first_zero - 1
        except ValueError:
            last_non_pad_idx = len(attention_mask) - 1

        last_token_hiddens = hidden_states[last_non_pad_idx]

        for layer_idx, layer_hidden in enumerate(last_token_hiddens):
            for col in columns_to_select:
                result[col].append(row[col])
            result["layer"].append(layer_idx)
            result["hidden_state"].append(layer_hidden)

    return pd.DataFrame(result)


def make_word_level_probing_dataset(
        model_name: str,
        model_hiddens_path: Path,
        stimuli_csv_path: Path,
        hiddens_results_csv_output_name: str = "results.csv",
        instruct: bool=False) -> pd.DataFrame:

    hiddens_results_path = model_hiddens_path / hiddens_results_csv_output_name

    results_df = pd.read_csv(hiddens_results_path)
    stimuli_df = pd.read_csv(stimuli_csv_path)

    merged_df = pd.merge(results_df, stimuli_df, how="inner", on="sentence")

    assert merged_df.shape[0] == stimuli_df.shape[0]

    columns_to_select = [
        "sentence_id", "sentence", "congruent", "structure", "length", "position", "target"
    ]

    merged_df = merged_df[columns_to_select + ["hidden_states_path"]]

    result = {col: [] for col in columns_to_select}

    result = {**result, **{"layer": [], "hidden_state": []}}
    tokenizer = load_tokenizer(model_name)

    tokenized_sentences = tokenize(
        tokenizer=tokenizer,
        sentences=merged_df["sentence"].tolist(),
        use_chat_template=instruct,
    )

    for i, row in merged_df.iterrows():

        hidden_states_path = row["hidden_states_path"]
        hidden_states = np.load(os.path.abspath(os.path.join(str(model_hiddens_path.parent.parent), hidden_states_path.split(".npy")[0] + "_word_level_hidden_states.npy"))) # hidden shape [layers, emb_dim]

        for layer_idx, layer_hidden in enumerate(hidden_states):
            for col in columns_to_select:
                result[col].append(row[col])
            result["layer"].append(layer_idx)
            result["hidden_state"].append(layer_hidden)

    return pd.DataFrame(result)



if __name__ == "__main__":
    print(make_sentence_level_probing_dataset(
        model_name="ai-sage/GigaChat3-10B-A1.8B-base",
        model_hiddens_path=pathlib.Path("/Users/ilyamikheev/Desktop/cuscience/SIGNAL-analysis/src/results/ai-sage_GigaChat3-10B-A1.8B-base"),
        stimuli_csv_path=pathlib.Path("/Users/ilyamikheev/Desktop/cuscience/SIGNAL-analysis/hf_datasets/stimuli.csv")
    ).to_csv("./gen.csv", index=False))