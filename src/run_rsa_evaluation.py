import numpy as np
import os

import json

from rsa import windowed_rsa
from tqdm import tqdm

if __name__ == "__main__":

    erp_path = "../erp/erp_data.npy"
    erp = np.load(erp_path)

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


        rsa_results: dict[int, dict] = dict()

        for file_index, matrix in tqdm(matrices.items(), desc="Рассчёт RSA"):

            result = windowed_rsa(
                eeg_data=erp,
                neural_representations=matrix,
                window_size=100,
                step_size=50,
                sfreq=1_000,
                tmin=-0.4,
            )

            rsa_results[file_index] = result

        json.dump(rsa_results, open(matrix_path + "/" + "rsa_json" + ".json", "w"))