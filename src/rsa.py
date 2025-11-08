import numpy as np

from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, spearmanr, pearsonr


def rsa_v2(X: np.ndarray, Y: np.ndarray) -> (np.ndarray, float):
    """
    X: np.ndarray of shape (n_samples, n_features)
    """
    X_norm = zscore(X, axis=0, ddof=1)
    condensed = pdist(X_norm, metric="cosine")
    X_rdm = squareform(condensed)

    Y_norm = zscore(Y, axis=0, ddof=1)
    condensed = pdist(Y_norm, metric="cosine")
    Y_rdm = squareform(condensed)

    idx = np.triu_indices_from(X_rdm, k=1)
    rsa_score, pval = spearmanr(X_rdm[idx], Y_rdm[idx])

    return rsa_score, pval


def get_times_from_eeg_data(eeg_data: np.ndarray,
                            sfreq=1_000,
                            tmin=-0.4) -> np.ndarray:

    n_times = eeg_data.shape[2]
    duration = n_times / sfreq
    tmax = tmin + duration
    times = np.linspace(tmin, tmax, n_times, endpoint=False)

    return times

def windowed_rsa(
        eeg_data: np.ndarray,
        neural_representations: np.ndarray,
        window_size: int,
        step_size: int,
        sfreq: int = 1_000,
        tmin: float = -0.4,
        method: str = "spearman",
        compute_permutations: bool = False,
        n_permutations: int = 5_000,
):

    if eeg_data.shape[0] != neural_representations.shape[0]:
        raise RuntimeError(f"Stimuli's dimensions do not match")

    n_times = eeg_data.shape[2]
    times = get_times_from_eeg_data(eeg_data, sfreq=sfreq, tmin=tmin)

    neural_condensed = pdist(neural_representations, metric="cosine")
    neural_rdm = squareform(neural_condensed)

    window_correlations = []
    window_p_values = []
    window_centers = []
    window_perm_distributions = []

    for start_idx in range(0, n_times - window_size + 1, step_size):

        end_idx = start_idx + window_size
        window_center = (times[start_idx] + times[end_idx - 1]) / 2

        eeg_window = eeg_data[:, :, start_idx:end_idx].mean(axis=2)
        eeg_window_condensed = pdist(eeg_window, metric="cosine")
        eeg_rdm = squareform(eeg_window_condensed)

        triu_indices = np.triu_indices_from(eeg_rdm, k=1)
        eeg_triu = eeg_rdm[triu_indices]
        neural_triu = neural_rdm[triu_indices]

        if method == "spearman":
            correlation, p_value = spearmanr(eeg_triu, neural_triu)
        elif method == "pearson":
            correlation, p_value = pearsonr(eeg_triu, neural_triu)
        else:
            raise RuntimeError(f"Method {method} not supported")

        # Перестановочный тест

        if compute_permutations:

            perm_correlations = []

            for permutation in range(n_permutations):
                n_sentences = eeg_data.shape[0]
                shuffled_indices = np.random.permutation(n_sentences)

                eeg_shuffled = eeg_window[shuffled_indices]
                eeg_shuffled_condensed = pdist(eeg_shuffled, metric="cosine")
                eeg_shuffled_rdm = squareform(eeg_shuffled_condensed)

                shuffled_triu_indices = np.triu_indices_from(eeg_shuffled_rdm, k=1)
                eeg_shuffled_triu = eeg_shuffled_rdm[shuffled_triu_indices]

                if method == "spearman":
                    shuffled_corr, _ = spearmanr(eeg_shuffled_triu, neural_triu)
                elif method == "pearson":
                    shuffled_corr, _ = pearsonr(eeg_shuffled_triu, neural_triu)
                else:
                    raise RuntimeError(f"Method {method} not supported")
                perm_correlations.append(shuffled_corr)

            perm_correlations = np.array(perm_correlations)
            p_value = np.mean(np.abs(perm_correlations) >= np.abs(correlation))
            window_perm_distributions.append(perm_correlations)

        window_correlations.append(correlation)
        window_p_values.append(p_value)
        window_centers.append(window_center)

    return {
        'correlations': np.array(window_correlations).tolist(),
        'p_values': np.array(window_p_values).tolist(),
        'window_centers': np.array(window_centers).tolist(),
        'window_perm_distributions': np.array(window_perm_distributions).tolist()
    }



def fixed_window_rsa(
        eeg_data: np.ndarray,
        neural_representations: np.ndarray,
        window_ms: int,
        step_ms: int,
        sfreq: int = 1_000,
):

    window_size = int(window_ms * sfreq / 1000)
    step_size = int(step_ms * sfreq / 1000)

    return windowed_rsa(eeg_data, neural_representations, window_size, step_size)
