#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author(s): Jan Svec
# @Email(s): isvecjan@fit.vutbr.cz


import os
import sys
import argparse
import logging
from pathlib import Path

import glob

import numpy as np
import pandas as pd
import soundfile as sf
from os.path import basename

from scipy.signal import resample_poly

from mix import (
    get_loudness, 
    mix, 
    check_for_cliping, 
    compute_gain, 
    mix_w_gain
)

from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SEED = 1234
REPEAT_RUN = 3
MIN_LENGTH = 84500
N_INTERFERENCE = 3 * 192 # 576
PREFIX_NAME = "03-01-"
SAMPLE_RATE = 8000
N_RAVDESS_SPEAKERS = 25

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

emotional_intensity = {
    "01": "normal",
    "02": "strong",
}

statement = {
    "01": "utt1",
    "02": "utt2",
}

repetition = {
    "01": "1time",
    "02": "2time",
}


def main(args):
    """
    Main function
    """
    logger.info(" Prepare RAVDESS2Mix ... starting")

    librispeech_md = pd.read_csv(args.librispeech_test_meta_file, engine='python')
    librispeech_md = librispeech_md[librispeech_md['length'] > MIN_LENGTH]
    emo = ["01", "02", "03", "04", "05", "06", "07", "08",] # ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    intensity = ["01", "02"] # ['normal', 'strong']
    utts = ["01", "02"] # [utt1', 'utt2']
    times = ["01", "02"] # ['1time', '2time']
    spk_ids = list(range(1, N_RAVDESS_SPEAKERS ))

    def utt_revert(utt):
        return "02" if utt == "01" else "01"

    interf_spk_files = [i[4] for i in librispeech_md.sample(N_INTERFERENCE, random_state=_SEED).values]

    i, j = 0, 0
    for intnsty in intensity:
        mix_dir = os.path.join(args.ravdess2mix_dir, emotional_intensity[intnsty], "mix")
        target_dir = os.path.join(args.ravdess2mix_dir, emotional_intensity[intnsty], "target")
        interfer_dir = os.path.join(args.ravdess2mix_dir, emotional_intensity[intnsty], "interference")
        enroll_dir = os.path.join(args.ravdess2mix_dir, emotional_intensity[intnsty], "enroll")
        speech_extraction_csv_dir = os.path.join(args.ravdess2mix_dir, emotional_intensity[intnsty], "csv_extraction")
        speech_separation_csv_dir = os.path.join(args.ravdess2mix_dir, emotional_intensity[intnsty], "csv_separation")

        os.makedirs(mix_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(interfer_dir, exist_ok=True)
        os.makedirs(enroll_dir, exist_ok=True)
        os.makedirs(speech_extraction_csv_dir, exist_ok=True)
        os.makedirs(speech_separation_csv_dir, exist_ok=True)
        for e1 in emo:
            flag_sep = False
            if os.path.isfile(os.path.join(speech_separation_csv_dir, f"{emotions[e1]}.csv")):
                flag_sep = True
            for e2 in emo:
                if intnsty == "02" and (e1 == "01" or e2 == "01"):
                    continue
                with open(os.path.join(speech_extraction_csv_dir, f"{emotions[e1]}_{emotions[e2]}.csv"), "w") as fd_ex, \
                     open(os.path.join(speech_separation_csv_dir, f"{emotions[e1]}.csv"), "a") as fd_sep:
                    print("mixture_ID,mixture_path,auxiliary_path,source_path,noise_path,length", file=fd_ex)
                    if not flag_sep:
                        print("mixture_ID,mixture_path,source_1_path,source_2_path,noise_path,length", file=fd_sep)
                    for _ in range(REPEAT_RUN):
                        for spk_id in spk_ids:
                            for utt in utts:
                                for tm1 in times:
                                    for tm2 in times:
                                        tgt_label = f"{PREFIX_NAME}{e1}-{intnsty}-{utt}-{tm1}-{spk_id:02d}"
                                        interfer_label = f"{basename(interf_spk_files[i])[:-5]}"
                                        mixture_label = f"{tgt_label}_{interfer_label}"

                                        enroll_label = f"{PREFIX_NAME}{e2}-{intnsty}-{utt_revert(utt)}-{tm2}-{spk_id:02d}"
                                        mixture_filename = os.path.join(mix_dir, f"{mixture_label}.wav")
                                        enroll_filename = os.path.join(enroll_dir, f"{enroll_label}.wav")
                                        target_filename = os.path.join(target_dir, f"{mixture_label}.wav")
                                        interfer_filename = os.path.join(interfer_dir, f"{mixture_label}.wav")

                                        trg, sr = sf.read(os.path.join(args.ravdess_dataset_dir, f"Actor_{spk_id:02d}/{tgt_label}.wav"))
                                        if len(trg.shape) == 2:
                                            trg = trg[:, 0]
                                        if sr != SAMPLE_RATE:
                                            trg = resample_poly(trg, SAMPLE_RATE, sr)

                                        nontrg, sr = sf.read(os.path.join(args.librispeech_dir, interf_spk_files[i]))
                                        if len(nontrg.shape) == 2:
                                            nontrg = nontrg[:, 0]
                                        if sr != SAMPLE_RATE:
                                            nontrg = resample_poly(nontrg, SAMPLE_RATE, sr)

                                        aux, sr = sf.read(os.path.join(args.ravdess_dataset_dir, f"Actor_{spk_id:02d}/{enroll_label}.wav"))
                                        if len(aux.shape) == 2:
                                            aux = aux[:, 0]
                                        if sr != SAMPLE_RATE:
                                            aux = resample_poly(aux, SAMPLE_RATE, sr)

                                        length = min([d.shape[0] for d in [trg, nontrg, aux]])
                                        loudness, _, sources_list_norm = get_loudness([trg[:length], nontrg[:length]])
                                        mixture = mix(sources_list_norm)
                                        renormalize_loudness, _ = check_for_cliping(mixture, sources_list_norm)
                                        gain_list = compute_gain(loudness, renormalize_loudness)
                                        mixture = mix_w_gain([trg[:length], nontrg[:length]], gain_list)
                                        trg = trg[:length]
                                        nontrg =  nontrg[:length]

                                        if not os.path.isfile(mixture_filename):
                                            sf.write(mixture_filename, mixture, SAMPLE_RATE)
                                        if not os.path.isfile(interfer_filename):
                                            sf.write(interfer_filename, nontrg, SAMPLE_RATE)
                                        if not os.path.isfile(target_filename):
                                            sf.write(target_filename, trg, SAMPLE_RATE)
                                        if not os.path.isfile(enroll_filename):
                                            sf.write(enroll_filename, aux, SAMPLE_RATE)

                                        print(f"{mixture_label},{mixture_filename},{enroll_filename},{target_filename},xxx,{mixture.shape[0]}",file=fd_ex)
                                        if not flag_sep:
                                            print(f"{mixture_label},{mixture_filename},{target_filename},{interfer_filename},xxx,{mixture.shape[0]}",file=fd_sep)

                                        i += 1
                                        j += 1
                    i=0
    logger.info(" Prepare RAVDESS2Mix ... ending")


if __name__ == '__main__':
    np.random.seed(_SEED)
    os.environ['PYTHONHASHSEED'] = str(_SEED)

    parser = argparse.ArgumentParser(description='Prepare RAVDESS2Mix')
    parser.add_argument("--librispeech-dir", "-librispeech-dir",
                        type=str, default='/mnt/data/LibriSpeech',
                        help='LibriSpeech directory.')
    parser.add_argument("--librispeech-test-meta-file", "-librispeech-test-meta-file", 
                        type=str, default='/mnt/data/LibriSpeech',
                        help='LibriSpeech directory.')
    parser.add_argument("--ravdess-dataset-dir", "-ravdess-dataset-dir",
                        type=str, default='/mnt/data/RAVDESS',
                        help='RAVDESS directory. Directory should contain subdirectories for each actor (Actor_*).')
    parser.add_argument("--ravdess2mix-dir", "-ravdess2mix-dir",
                        type=str, default='test-clean.meta.csv',
                        help="Metafile for LibriSpeech's clean testset")
    args = parser.parse_args()

    sys.exit(main(args))
