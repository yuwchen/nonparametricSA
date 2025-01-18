import argparse
import os

import faiss
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel
from utils import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rootdir",
        type=str,
        default="./data/coswara/",
        help="Path of wavefile directory",
    )
    parser.add_argument(
        "--traintxt",
        type=str,
        default="./txtfile/coswara-train-normal-2-f.txt",
        help="Path of training text file",
    )

    args = parser.parse_args()

    rootdir = args.rootdir
    train_txt = args.traintxt

    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    num_of_layer = 24
    feature_dim = 1024
    mode = "hubert-large-ft"

    model = model.to("cuda")
    model.eval()

    # load training data list
    train_df = open_info(train_txt)

    """
    create database
    """
    output_df = {}
    for layer_idx in range(num_of_layer):
        output_df[layer_idx] = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):

        filepath = row["filename"]

        the_sex = row["Sex"]
        the_age = row["Age"]
        the_label = row["label"]

        wavpath = os.path.join(rootdir, filepath)

        wav, sr = torchaudio.load(wavpath)
        wav = process_wav(wav, sr)

        with torch.no_grad():

            input_values = processor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values
            input_values = input_values.to("cuda")
            x = model(input_values, output_hidden_states=True).hidden_states

        for layer_idx in range(num_of_layer):

            embed = x[layer_idx]
            embed = embed.detach().cpu().squeeze(0)
            embed = torch.mean(embed, 0).numpy()
            output_df[layer_idx].append(
                {
                    "filename": filepath,
                    "label": the_label,
                    "feature": embed,
                    "sex": the_sex,
                    "age": the_age,
                },
            )

    for layer_idx, value in output_df.items():

        train_df_uttr = pd.DataFrame(value)
        faiss_uttr = faiss.IndexFlatL2(feature_dim)
        xb = np.asarray(train_df_uttr["feature"].to_list())
        print("Create db for layer {}, db data shape:{}".format(layer_idx, xb.shape))

        faiss_uttr.add(xb)
        train_df_uttr = train_df_uttr.drop(columns=["feature"])
        outputdir = os.path.join("faiss_database", "{}".format(mode))
        creat_dir(outputdir)

        # save additional information in the .csv file
        train_df_uttr.to_csv(os.path.join(outputdir, "{}.csv".format(layer_idx)))
        # write the db
        faiss.write_index(
            faiss_uttr, os.path.join(outputdir, "{}.index".format(layer_idx))
        )

    """
    create database using reversed wav
    """
    output_df_re = {}
    for layer_idx in range(num_of_layer):
        output_df_re[layer_idx] = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):

        filepath = row["filename"]

        the_sex = row["Sex"]
        the_age = row["Age"]
        the_label = row["label"]

        wavpath = os.path.join(rootdir, filepath)

        wav, sr = torchaudio.load(wavpath)
        wav = process_wav(wav, sr)
        wav = wav.flip(dims=[0])

        with torch.no_grad():
            input_values = processor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values
            input_values = input_values.to("cuda")
            x = model(input_values, output_hidden_states=True).hidden_states

        for layer_idx in range(num_of_layer):

            embed = x[layer_idx]
            embed = embed.detach().cpu().squeeze(0)
            embed = torch.mean(embed, 0).numpy()

            output_df_re[layer_idx].append(
                {
                    "filename": filepath,
                    "label": the_label,
                    "feature": embed,
                    "sex": the_sex,
                    "age": the_age,
                },
            )

    for layer_idx, value in output_df_re.items():

        train_df_uttr = pd.DataFrame(value)
        faiss_uttr = faiss.IndexFlatL2(feature_dim)
        xb = np.asarray(train_df_uttr["feature"].to_list())
        print("Create db for layer {}, db data shape:{}".format(layer_idx, xb.shape))
        faiss_uttr.add(xb)
        train_df_uttr = train_df_uttr.drop(columns=["feature"])
        outputdir = os.path.join("faiss_database", "{}-re".format(mode))
        creat_dir(outputdir)
        train_df_uttr.to_csv(os.path.join(outputdir, "{}.csv".format(layer_idx)))
        faiss.write_index(
            faiss_uttr, os.path.join(outputdir, "{}.index".format(layer_idx))
        )


if __name__ == "__main__":
    main()
