import os
import csv
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

TokenSelector = Union[int, Dict[int, int], Callable[[np.ndarray, dict], int]]

@dataclass
class BuildOptions:
    metric: str = "cosine"          # 'cosine' | 'correlation' | 'euclidean' | 'spearman'
    zscore_features: bool = True
    return_square: bool = True
    save_dir: Optional[str] = None
    save_dtype: str = "float32"
    spearman_ties_method: str = "average"
    layers: Optional[Iterable[int]] = None
    conditions_col: Optional[str] = None


def _default_token_selector(hidden: np.ndarray, row: dict) -> int:
    return hidden.shape[1] - 1

def resolve_selector(selector: Optional[TokenSelector]) -> Callable[[np.ndarray, dict, int], int]:
    if selector is None:
        def f(hidden, row, row_idx): return _default_token_selector(hidden, row)
        return f
    if isinstance(selector, int):
        def f(hidden, row, row_idx): return int(selector)
        return f
    if isinstance(selector, dict):
        def f(hidden, row, row_idx): return int(selector[row_idx])
        return f
    if callable(selector):
        def f(hidden, row, row_idx): return int(selector(hidden, row))
        return f
    raise ValueError("Unsupported token selector")


def build_feature_matrices(
    rows: List[dict],
    token_selector: Optional[TokenSelector] = None,
    layers: Optional[Iterable[int]] = None,
) -> Dict[int, np.ndarray]:
    sel = resolve_selector(token_selector)

    first = np.load(rows[0]["hidden_states_path"])
    if first.ndim != 3:
        raise ValueError("Ожидается массив [n_layers, token_num, emb_dim]")
    n_layers, _, emb_dim = first.shape

    target_layers = list(layers) if layers is not None else list(range(n_layers))
    L = len(target_layers)
    N = len(rows)

    X_by_layer = {l: np.empty((N, emb_dim), dtype=first.dtype) for l in target_layers}

    for i, row in enumerate(rows):
        h = np.load(row["hidden_states_path"])  # [L, T, D]
        if h.ndim != 3:
            raise ValueError(f"{row['hidden_states_path']} имеет неверную размерность")
        t = sel(h, row, i)
        if not (0 <= t < h.shape[1]):
            raise IndexError(f"token_index {t} вне диапазона [0,{h.shape[1]-1}] для строки {i}")
        for l in target_layers:
            X_by_layer[l][i, :] = h[l, t, :]

    return X_by_layer


def _prepare_for_metric(X: np.ndarray, metric: str, spearman_ties_method: str) -> np.ndarray:
    if metric.lower() == "spearman":
        from scipy.stats import rankdata
        Xr = np.empty_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            Xr[i, :] = rankdata(X[i, :], method=spearman_ties_method)
        return Xr
    return X


def compute_rdm_from_features(
    X: np.ndarray,
    metric: str = "cosine",
    zscore_features: bool = True,
    return_square: bool = True,
    spearman_ties_method: str = "average",
) -> np.ndarray:

    X_proc = X.astype(np.float64, copy=False)
    if zscore_features:
        X_proc = zscore(X_proc, axis=0, ddof=1)  # по столбцам

    metric = metric.lower()
    if metric == "spearman":

        X_proc = _prepare_for_metric(X_proc, metric, spearman_ties_method)
        condensed = pdist(X_proc, metric="correlation")
    elif metric in ("cosine", "correlation", "euclidean"):
        condensed = pdist(X_proc, metric=metric)
    else:
        raise ValueError(f"Неизвестная метрика: {metric}")

    if return_square:
        return squareform(condensed)
    return condensed


def split_indices_by_condition(rows: List[dict], conditions_col: str) -> Dict[str, List[int]]:
    if conditions_col not in rows[0]:
        raise ValueError(f"В CSV нет колонки '{conditions_col}', а она нужна для внутриусловных RDM")
    buckets: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        key = str(r[conditions_col])
        buckets.setdefault(key, []).append(i)
    return buckets


def slice_rdm(rdm: np.ndarray, indices: List[int]) -> np.ndarray:
    idx = np.array(indices, dtype=int)
    return rdm[np.ix_(idx, idx)]


def build_rdms_for_all_layers(
    csv_path: str,
    token_selector: Optional[TokenSelector] = None,
    opts: Optional[BuildOptions] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Dict[int, np.ndarray]]]:

    if opts is None:
        opts = BuildOptions()
    rows = read_metadata_csv(csv_path)

    X_by_layer = build_feature_matrices(
        rows,
        token_selector=token_selector,
        layers=opts.layers,
    )


    full_rdms: Dict[int, np.ndarray] = {}
    for l, X in X_by_layer.items():
        R = compute_rdm_from_features(
            X,
            metric=opts.metric,
            zscore_features=opts.zscore_features,
            return_square=True,
            spearman_ties_method=opts.spearman_ties_method,
        ).astype(opts.save_dtype, copy=False)
        full_rdms[l] = R
        if opts.save_dir:
            os.makedirs(opts.save_dir, exist_ok=True)
            np.save(os.path.join(opts.save_dir, f"RDM_full_layer{l}.npy"), R)


    within_rdms: Dict[str, Dict[int, np.ndarray]] = {}
    if opts.conditions_col:
        buckets = split_indices_by_condition(rows, opts.conditions_col)
        for cond, idxs in buckets.items():
            within_rdms[cond] = {}
            for l, R in full_rdms.items():
                Rc = slice_rdm(R, idxs).astype(opts.save_dtype, copy=False)
                within_rdms[cond][l] = Rc
                if opts.save_dir:
                    np.save(os.path.join(opts.save_dir, f"RDM_{cond}_layer{l}.npy"), Rc)

    return full_rdms, within_rdms
