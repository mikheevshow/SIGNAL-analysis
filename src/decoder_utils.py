
from typing import List, Dict, Literal, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

AggT = Literal["offset", "onset", "mean", "sum"]

def find_word_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for m in re.finditer(r"\w+", text, flags=re.UNICODE):
        spans.append((m.start(), m.end(), m.group(0)))
    return spans

def map_tokens_to_words(offset_mapping, word_spans) -> List[int]:
    """
    На каждый токен отдает индекс слова (или -1, если токен — специальный/пробел).
    Сопоставление по перекрытию по символам.
    """
    token_to_word = []
    wi = 0
    for (tok_start, tok_end) in offset_mapping:
        if tok_start == tok_end:
            token_to_word.append(-1)
            continue
        while wi < len(word_spans) and word_spans[wi][1] <= tok_start:
            wi += 1
        if wi < len(word_spans):
            w_start, w_end, _ = word_spans[wi]
            overlap = max(0, min(tok_end, w_end) - max(tok_start, w_start))
            token_to_word.append(wi if overlap > 0 else -1)
        else:
            token_to_word.append(-1)
    return token_to_word

@torch.no_grad()
def extract_word_hidden_states(
    texts: List[str],
    model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    agg: AggT = "offset",
    layers: Literal["all", "last"] = "all",
    fp16: bool = False,
    max_length: int = 2048,
) -> List[Dict]:
    """
    Для каждого исходного текста возвращает dict с:
      - 'words': список слов (str)
      - 'word_spans': [(start,end)]
      - 'token_ids': ids с учётом спец-токенов
      - 'token_to_word': список индексов слова для каждого токена
      - 'hidden_by_word': список длины n_words, где элемент —
            torch.Tensor [n_layers, hidden_size] (или [1, hidden] если layers='last')
        — агрегированное представление слова по выбранной стратегии.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if hasattr(tok, "add_prefix_space"):
        tok.add_prefix_space = True

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval().to(device)
    if fp16 and device == "cuda":
        model = model.half()

    results = []
    for text in texts:
        word_spans = find_word_spans(text)
        enc = tok(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        token_to_word = map_tokens_to_words(offsets, word_spans)

        out = model(**enc.to(device), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        hs = hs[1:]
        H = torch.stack([h[0] for h in hs], dim=0)
        if layers == "last":
            H = H[-1:].contiguous()

        n_layers, T, Hdim = H.shape

        n_words = len(word_spans)
        token_idx_by_word = [[] for _ in range(n_words)]
        for t, w in enumerate(token_to_word):
            if w >= 0 and attn_mask[0, t].item() == 1:
                token_idx_by_word[w].append(t)

        hidden_by_word: List[torch.Tensor] = []
        for wi, tids in enumerate(token_idx_by_word):
            if not tids:
                hidden_by_word.append(torch.full((1 if layers=="last" else n_layers, Hdim), float('nan'), device=device))
                continue

            if agg == "offset":
                idx = tids[-1]
                vec = H[:, idx, :]
            elif agg == "onset":
                idx = tids[0]
                vec = H[:, idx, :]
            elif agg == "mean":
                vec = H[:, tids, :].mean(dim=1)
            elif agg == "sum":
                vec = H[:, tids, :].sum(dim=1)
            else:
                raise ValueError(f"Unknown agg: {agg}")

            hidden_by_word.append(vec.detach().cpu())

        results.append(dict(
            text=text,
            words=[w for *_, w in word_spans],
            word_spans=[(a, b) for a, b, _ in word_spans],
            token_ids=enc["input_ids"][0].detach().cpu().tolist(),
            token_to_word=token_to_word,
            hidden_by_word=hidden_by_word,
            n_layers=(1 if layers=="last" else n_layers),
            hidden_size=Hdim,
            agg=agg,
            model_name=model_name
        ))
    return results

if __name__ == "__main__":
    texts = [
        "Вчера студентка читала интересную статью по нейролингвистике.",
        "Он идут в магазин за хлебом."  
    ]
    out = extract_word_hidden_states(
        texts,
        model_name="gpt2",
        agg="offset",
        layers="all",
        fp16=False
    )

    sample = out[0]
    word_idx = 2
    layer_idx = 5
    vec = sample["hidden_by_word"][word_idx][layer_idx]
    print("Word:", sample["words"][word_idx])
    print("Vector shape:", tuple(vec.shape))
