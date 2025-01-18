import os

import pandas as pd
import torch
import torchaudio
from sklearn.metrics import roc_auc_score


def creat_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def process_wav(wav, sr):

    if wav.shape[0] != 1:
        wav = torch.mean(wav, 0)
        wav = torch.reshape(wav, (1, -1))

    if sr != 16000:
        transform = torchaudio.transforms.Resample(sr, 16000)
        wav = transform(wav)

    wav = torch.reshape(wav, (-1,))
    return wav


def parse_line(line):

    split_data = line.strip().split(";")
    wavfile = split_data[0]
    metadata = {}

    for item in split_data[2:]:
        if ":" in item:
            key, value = item.split(":")
            metadata[key] = value

    metadata["filename"] = wavfile
    metadata["label"] = int(split_data[1])

    return metadata


def open_info(file_path):
    rows = []
    with open(file_path, "r") as file:
        for line in file:
            parsed_row = parse_line(line)
            rows.append(parsed_row)

    samples = pd.DataFrame(rows)
    return samples


def calculate_scores(gt, pred):
    tp = tn = fp = fn = 0

    for a, p in zip(gt, pred):
        if a == 1 and p == 1:
            tp += 1
        elif a == 0 and p == 0:
            tn += 1
        elif a == 0 and p == 1:
            fp += 1
        elif a == 1 and p == 0:
            fn += 1

    return tp, tn, fp, fn


def evaluate(gt, pred):

    tp, tn, fp, fn = calculate_scores(gt, pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return (
        round(roc_auc_score(gt, pred), 3),
        round(sensitivity, 3),
        round(specificity, 3),
    )
