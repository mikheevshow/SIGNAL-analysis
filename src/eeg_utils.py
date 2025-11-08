import os
import mne
import pandas as pd
import numpy as np

from mne import Epochs


# С датасетом ContributorsSIGNAL/SIGNAL есть проблема
# 1) В файлах эпох лежат данные для испытуемых с отфильтрованными стимулами (<= 600 записей)
# 2) В файлах информациях об эпохах лежат данные обо всех стимулах (600) и их временные метки
# 3) Временные метки из .fif файлов не совпадают с метками из .csv

# Я взял датасет из черновика статьи zhuravlevahana/SIGNAL
# 1) для первых двух пациентов epoch_events совпали с epoch_info по колонке start
# 2) для остальных по колонке time2
# 3) это позволило получить связь с тем какие предложения остались в ЭЭГ после фильтрации, для дальнейшего анализа (например построения RSA)

### Видимо, формат файлов epoch_info для чернового варианта набора данных менялся, а вот .fif файлы
### похоже переносились без изменений, поэтому сейчас итоговоый набор данных не совсем валиден

def align_events(dataset_folder: str, aligned_files_output: str) -> None:
    os.makedirs(aligned_files_output, exist_ok=True)

    xlsx_files = []
    fif_files = []
    for file_name in os.listdir(dataset_folder):
        if str(file_name).endswith(".xlsx"):
            xlsx_files.append(str(os.path.join(dataset_folder, file_name)))
        if str(file_name).endswith(".fif"):
            fif_files.append(str(os.path.join(dataset_folder, file_name)))

    fif_p_dict: dict[str, str] = {}
    for file_name in fif_files:
        fif_p_name = file_name.split("/")[-1].split("_")[0]
        fif_p_dict[fif_p_name] = file_name

    for xlsx_file in xlsx_files:
        xlsx_p_name = xlsx_file.split("/")[-1].split("_")[0]
        fif_file = fif_p_dict[xlsx_p_name]

        print(xlsx_file, fif_file)

        if xlsx_p_name in {"p0", "p1"}:
            col = "start"
        else:
            col = "time2"

        xlsx_df = pd.read_excel(xlsx_file)
        from ast import literal_eval
        xlsx_df["sentence_combined"] = xlsx_df["sentence"].apply(lambda x: " ".join(literal_eval(x)))

        epoch = mne.read_epochs(fif_file, preload=True)
        event_id_dict = dict(epoch.event_id)
        epochs_events = pd.DataFrame(epoch.events, columns=[col, "1", "event_id"])

        try:
            merged_df = epochs_events.merge(xlsx_df, how="left", on=col)
        except Exception as e:
            print(e)
            print(xlsx_file)
            break

        for i, row in merged_df.iterrows():
            stimuli_id = row["event_id"]
            stimuli_name = row["name"]
            if event_id_dict[stimuli_name] != int(stimuli_id):
                print(row)
                raise RuntimeError("Stimuli are not equal!")

        merged_df = merged_df.rename(columns={col: "event_time"})
        merged_df.to_csv(aligned_files_output + "/" + f"{xlsx_p_name}_aligned.csv", index=False)


def get_eeg_for_all_sentences(
    sentences: list[str],
    dataset_dir: str,
    aligned_files_dir: str,
):

    preloaded_epochs_dict: dict[str, Epochs] = dict()

    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".fif"):
            epoch_name = file_name.split("_")[0]
            preloaded_epochs_dict[epoch_name] = mne.read_epochs(dataset_dir + "/" + file_name)


    aligned_df_dict: dict[str, pd.DataFrame] = dict()

    for file_name in os.listdir(aligned_files_dir):
        if file_name.endswith(".csv"):
            df_epoch_name = file_name.split("_")[0]
            aligned_df_dict[df_epoch_name] = pd.read_csv(aligned_files_dir + "/" + file_name)

    epochs_names = [f"p{i}" for i in range(len(aligned_df_dict.keys()))]

    arrs = []

    # Тут должен получиться массив sentences x channel x n_time
    for epoch_name in epochs_names:

        ep: Epochs = preloaded_epochs_dict[epoch_name]
        df: pd.DataFrame = aligned_df_dict[epoch_name]
        idx = df.set_index("sentence_combined").index.get_indexer(sentences)

        data = np.full(
            (len(sentences), ep.get_data().shape[1], ep.get_data().shape[2]),
            np.nan,
            dtype=float
        )

        mask = idx != -1

        data[mask] = ep.get_data()[idx[mask]]

        arrs.append(data)

    all_data = np.stack(arrs, axis=0)

    return all_data


def average_over_subjects_with_info(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """
    Усредняет данные ЭЭГ по испытуемым и возвращает дополнительную информацию.

    Parameters:
    -----------
    data : np.ndarray
        Массив данных размерности: испытуемые x предложения x каналы x временные_точки

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        - erp_data: усредненный массив ERP размерности: предложения x каналы x временные_точки
        - subject_counts: количество испытуемых для каждого предложения (размерность: предложения)
    """

    subject_counts = np.sum(~np.isnan(data).all(axis=(2, 3)), axis=0)
    erp_data = np.nanmean(data, axis=0)

    return erp_data, subject_counts