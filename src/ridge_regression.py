import json
import os
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm


def average_nonoverlapping_windows(erp: np.ndarray, window: int):

    n_stim, n_ch, n_times = erp.shape
    n_windows = n_times // window

    cut_len = n_windows * window
    erp_cut = erp[:, :, :cut_len]

    erp_win = erp_cut.reshape(n_stim, n_ch, n_windows, window)

    erp_win = erp_win.mean(axis=3)

    return erp_win


def average_sliding_windows(erp: np.ndarray, window: int, step: int):

    n_stim, n_ch, n_times = erp.shape
    n_windows = (n_times - window) // step + 1

    erp_roll = np.zeros((n_stim, n_ch, n_windows))

    for i in range(n_windows):
        start = i * step
        end = start + window
        erp_roll[:, :, i] = erp[:, :, start:end].mean(axis=2)

    return erp_roll


def run_encoding_windows(erp_win: np.ndarray, hidden_states: list[np.ndarray], alpha:int=100.0, n_splits:int=5):

    n_layers = len(hidden_states)
    n_stim, emb_dim = hidden_states[0].shape
    _, n_ch, n_windows = erp_win.shape

    scores = np.zeros((len(hidden_states), n_ch, n_windows))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for i, layer_hiddens in tqdm(enumerate(hidden_states), desc="Running encoding windows"):
        X = layer_hiddens
        print(f"Layer {i+1}/{n_layers}")

        for ch in tqdm(range(n_ch), desc="Running channels"):
            for w in range(n_windows):
                y = erp_win[:, ch, w]

                fold_corrs = []
                for train_idx, test_idx in kf.split(X):
                    model = Ridge(alpha=alpha)
                    model.fit(X[train_idx], y[train_idx])
                    pred = model.predict(X[test_idx])

                    if np.std(pred) > 0 and np.std(y[test_idx]) > 0:
                        corr = np.corrcoef(pred, y[test_idx])[0, 1]
                    else:
                        corr = 0
                    fold_corrs.append(corr)

                scores[i, ch, w] = np.mean(fold_corrs)

    return scores


if __name__ == "__main__":

    erp_data = np.load("../erp/erp_data.npy")
    print(erp_data.shape)

    anolw = average_nonoverlapping_windows(erp_data, 100)

    print(anolw.shape)

    models = [
        "Qwen/Qwen2.5-7B"
    ]

    paths = [
        "./results/Qwen_Qwen2.5-7B/stimuli_matrix_by_layer"
    ]

    for model, matrix_path in zip(models, paths):

        # load npy
        matrices: dict[int, np.ndarray] = dict()
        for file in os.listdir(matrix_path):
            if file.endswith(".npy"):
                file_index = int(file.split(".")[0])
                matrices[file_index] = np.load(matrix_path + "/" + file)

        sorted_matrices = list(map(lambda x: x[1], sorted(matrices.items(), key=lambda x: x[0], reverse=False)))

        result = run_encoding_windows(
            erp_win=anolw,
            hidden_states=sorted_matrices
        )

        np.save(f"./eeg_ridge.npy", result)

        # import matplotlib.pyplot as plt
        # import numpy as np
        #
        # plt.imshow(result, cmap='hot', interpolation='nearest')
        # plt.show()