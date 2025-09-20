import torch
import logging
import logging_cofig

from datasets import load_dataset
from typing import List, Literal

from text_utils import find_index_of_incongruent_word
from tokenizer_utils import load_tokenizer, tokenize
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

AggT = Literal["offset", "onset", "mean", "sum"]

@torch.no_grad()
def _extract_word_hidden_states(
    texts: List[str],
    model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fp16: bool = False,
) -> list:

    tok = load_tokenizer(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval().to(device)
    if fp16 and device == "cuda":
        model = model.half()

    encoded_tokens = tokenize(tokenizer=tok, sentences=texts)

    encoded_tokens["input_ids"] = encoded_tokens["input_ids"].to(device)
    encoded_tokens["attention_mask"] = encoded_tokens["attention_mask"].to(device)

    with torch.inference_mode():
        output = model(input_ids=encoded_tokens["input_ids"],
                       attention_mask=encoded_tokens["attention_mask"],
                       output_hidden_states=True,
                       use_cache=False)

    hidden_states = torch.stack(output.hidden_states, dim=1)
    hidden_states = hidden_states.permute(0, 2, 1, 3)
    hidden_states = [hidden_states[j] for j in range(hidden_states.size(0))]

    results = []
    for t, hs, off_map in zip(texts, hidden_states, encoded_tokens["offset_mapping"]):
        results.append(dict({
            "text": t,
            "hidden_states": hs.cpu().detach(),
            "offset_mapping": off_map,
        }))

    return results


def calculate_aggregated_hidden_states(
    congruent_sentences: list[str],
    incongruent_sentences: list[str],
    model_name: str = "openai-community/gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fp16: bool = False,
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

if __name__ == "__main__":

    ds = load_dataset("ContributorsSIGNAL/SIGNAL")
    df = ds["train"].to_pandas()

    logger.info(df.columns)

    congruent_sentences = df["congruent"].tolist()
    incongruent_sentences = df["sentence"].tolist()

    outputs = calculate_aggregated_hidden_states(
        congruent_sentences=congruent_sentences,
        incongruent_sentences=incongruent_sentences,
        model_name="openai-community/gpt2",
        device="mps",
        fp16=False,
        aggregation_method="mean",
    )

    logger.info(outputs)