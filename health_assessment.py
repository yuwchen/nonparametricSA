import argparse
import os
from collections import Counter

import faiss
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.cluster import KMeans
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
        "--testtxt",
        type=str,
        default="./txtfile/coswara-test-normal-2-f.txt",
        help="Path of testing text file",
    )

    args = parser.parse_args()

    rootdir = args.rootdir
    test_txt = args.testtxt
    creat_dir("Results")
    output_path = os.path.join("Results", os.path.basename(test_txt).split(".")[0])

    h_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    h_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    h_model = h_model.to("cuda")
    h_model.eval()

    test_df = open_info(test_txt)

    h_layer_index = [3, 4, 5]
    h_layer_index_re = [3, 4, 5]

    # load db
    faiss_db_uttr = {}
    faiss_info_uttr = {}
    for l_idx in h_layer_index:
        path = os.path.join(
            "faiss_database", "hubert-large-ft", "{}.index".format(l_idx)
        )
        the_faiss_db = faiss.read_index(path)
        faiss_db_uttr[l_idx] = the_faiss_db
        the_faiss_info = pd.read_csv(path.replace(".index", ".csv"))
        faiss_info_uttr[l_idx] = the_faiss_info
        print(
            path,
            f"Number of elements in the db: {the_faiss_db.ntotal}",
            f"d: {the_faiss_db.d}",
            f"Number of row in the df: {len(the_faiss_info)}",
        )

    # load reversed db
    faiss_db_uttr_re = {}
    faiss_info_uttr_re = {}
    for l_idx in h_layer_index_re:
        path = os.path.join(
            "faiss_database", "hubert-large-ft-re", "{}.index".format(l_idx)
        )
        the_faiss_db = faiss.read_index(path)
        faiss_db_uttr_re[l_idx] = the_faiss_db
        the_faiss_info = pd.read_csv(path.replace(".index", ".csv"))
        faiss_info_uttr_re[l_idx] = the_faiss_info
        print(
            path,
            f"Number of elements in the db: {the_faiss_db.ntotal}",
            f"d: {the_faiss_db.d}",
            f"Number of row in the df: {len(the_faiss_info)}",
        )

    # retrieved top-n samples
    topk_list = [5]  # np.arange(1, 20 + 1, 1, dtype=int).tolist()

    # number of cluster for each layer
    n_cluster = {3: 2, 4: 2, 5: 2}

    for topk in topk_list:

        y_gt = []
        y_pred = []
        f_out = open(
            output_path + "-top-{}.txt".format(topk),
            "w",
        )

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

            filepath = row["filename"]
            the_label = row["label"]
            the_age = int(row["Age"])
            wavpath = os.path.join(rootdir, filepath)

            wav, sr = torchaudio.load(wavpath)
            wav = process_wav(wav, sr)

            input_values = h_processor(
                wav, return_tensors="pt", sampling_rate=16000
            ).input_values
            input_values = input_values.to("cuda")

            with torch.no_grad():
                x = h_model(input_values, output_hidden_states=True).hidden_states

            pred_layer = []  # collect all the retrieved labels

            # loop over all selected layer
            for l_idx in h_layer_index:

                embed = x[l_idx]
                embed = embed.detach().cpu().numpy().squeeze(0)
                retrieved_labels = []

                if (
                    embed.shape[0] < n_cluster[l_idx]
                ):  # if time step of the signal < n_cluster
                    kmeans_selected = KMeans(n_clusters=2)
                else:
                    kmeans_selected = KMeans(n_clusters=n_cluster[l_idx])

                # segment-level retrieval
                cluster_labels = kmeans_selected.fit_predict(embed)
                for c_label in np.arange(n_cluster[l_idx]):
                    frames_in_cluster = embed[cluster_labels == c_label]
                    mean_frame = np.mean(frames_in_cluster, axis=0).tolist()
                    mean_frame = np.reshape(mean_frame, (1, -1))
                    D, retrieved_idx = faiss_db_uttr[l_idx].search(mean_frame, topk)
                    retrieved_doc = faiss_info_uttr[l_idx].iloc[retrieved_idx[0]]
                    retrieved_labels.extend(retrieved_doc["label"].values.tolist())

                word_pred = dict(Counter(retrieved_labels))

                uttr_topk = len(retrieved_labels)
                the_feature_uttr = np.mean(embed, axis=0)
                the_feature_uttr = np.reshape(the_feature_uttr, (1, -1))

                # utterance-level retrieval
                D, retrieved_idx = faiss_db_uttr[l_idx].search(
                    the_feature_uttr, uttr_topk
                )
                retrieved_doc = faiss_info_uttr[l_idx].iloc[retrieved_idx[0]]

                # Profile-aware refinement was not applied to the Coswara dataset.
                """
                if the_age != 3:
                    filtered_retrieved_doc = retrieved_doc[
                        retrieved_doc["age"] == the_age
                    ]
                    if len(filtered_retrieved_doc) != 0:
                        retrieved_doc = filtered_retrieved_doc
                """

                uttr_pred = dict(Counter(retrieved_doc["label"].values.tolist()))

                retrieved_labels.extend(retrieved_doc["label"].values.tolist())

                # reversed-utterance-level retrieval
                D, retrieved_idx = faiss_db_uttr_re[l_idx].search(
                    the_feature_uttr, uttr_topk
                )
                retrieved_doc = faiss_info_uttr_re[l_idx].iloc[retrieved_idx[0]]

                """
                if the_age != 3:
                    filtered_retrieved_doc = retrieved_doc[
                        retrieved_doc["age"] == the_age
                    ]
                    if len(filtered_retrieved_doc) != 0:
                        retrieved_doc = filtered_retrieved_doc
                
                """

                reuttr_pred = dict(Counter(retrieved_doc["label"].values.tolist()))
                retrieved_labels.extend(retrieved_doc["label"].values.tolist())

                the_pred_l = Counter(retrieved_labels).most_common(1)[0][0]
                pred_layer.append(the_pred_l)
                f_out.write(
                    "filename:{};gt_label:{},l_idx:{};pred:{};w_pred:{};uttr_pred:{};reuttr_pred:{};\n".format(
                        filepath,
                        the_label,
                        l_idx,
                        the_pred_l,
                        word_pred,
                        uttr_pred,
                        reuttr_pred,
                    )
                )

            the_pred = Counter(pred_layer).most_common(1)[0][0]

            y_gt.append(the_label)
            y_pred.append(the_pred)

        roc, sen, spe = evaluate(y_gt, y_pred)
        print("topk:{}, ROC:{}, Sen:{}, Spe:{}".format(topk, roc, sen, spe))
        f_out.close()


if __name__ == "__main__":
    main()
