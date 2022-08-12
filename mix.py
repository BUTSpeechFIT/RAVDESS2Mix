
import os
import random
import warnings

import numpy
import torch
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

from typing import Union

# Global parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

MAX_INT16 = np.iinfo(np.int16).max


def normalize_audio(sig: Union[numpy.array, torch.tensor], sample_rate=8000, is_noise=False, output_type=torch.Tensor):
    MAX_AMP = 0.9
    MIN_LOUDNESS = -33
    MAX_LOUDNESS = -25

    meter = pyloudnorm.Meter(sample_rate)

    if isinstance(sig, torch.Tensor):
        sig = sig.numpy()
    elif isinstance(sig, numpy.ndarray):
        pass
    else:
        raise TypeError(f"Input variable sig has unsupported type: {type(sig)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_loudness = meter.integrated_loudness(sig)
        if is_noise:
            target_loudness = random.uniform(
                MIN_LOUDNESS - 5, MAX_LOUDNESS - 5
            )
        else:
            target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        sig = pyloudnorm.normalize.loudness(
            sig, c_loudness, target_loudness
        )

        # check for clipping
        if np.max(np.abs(sig)) >= 1:
            sig = sig * MAX_AMP / np.max(np.abs(sig))

    if output_type == torch.Tensor:
        return torch.from_numpy(sig)
    elif output_type == numpy.ndarray:
        return sig
    else:
        raise ValueError(f"output_type is undefined: {output_type}")


def get_loudness(sources_list):
    """ Compute original loudness and normalise them randomly """
    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Noise has a different loudness
        if i == len(sources_list) - 1:
            target_loudness = random.uniform(MIN_LOUDNESS - 5,
                                             MAX_LOUDNESS - 5)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def check_for_cliping(mixture_max, sources_list_norm):
    """Check the mixture (mode max) for clipping and re normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(RATE)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture_max))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip


def mix(sources_list):
    """ Mixture input sources """
    mixture = np.zeros_like(sources_list[0])
    for i in range(len(sources_list)):
        mixture += sources_list[i]
    return mixture


def mix_w_gain(sources_list, gain_list):
    """ Mixture input sources with gain """
    mixture = np.zeros_like(sources_list[0])
    # mixture = torch.zeros_like(sources_list[0])
    for i in range(len(sources_list)):
        mixture += (gain_list[i] * sources_list[i])
    return mixture


def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain between the original and target loudness """
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain


def resample_list(sources_list, freq_dest, freq_src=16000):
    """ Resample the source list to the desired frequency"""
    resampled_list = []
    # Resample each source
    for source in sources_list:
        resampled_list.append(resample_poly(source, freq_dest, freq_src))
    return resampled_list


if __name__ == "__main__":
    pass
