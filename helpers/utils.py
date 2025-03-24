import os
import librosa
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


def clip_bce(pred, target):
    """Binary crossentropy loss."""
    return F.binary_cross_entropy(pred, target)


def clip_ce(pred, target):
    return F.cross_entropy(pred, target)


def get_loss_func(loss_type):
    if loss_type == "clip_bce":
        return clip_bce
    if loss_type == "clip_ce":
        return clip_ce
    if loss_type == "asl_loss":
        loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
        return loss_func


def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [
        np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)
    ]
    return np.array(mixup_lambdas).astype(np.float32)


def extract_features(audio_path):
    """
    Extracts audio features from a given file to match the dataset used for training.
    """
    try:
        y, sr = librosa.load(audio_path, mono=True)

        if len(y) == 0:
            print(f"Warning: Empty file {audio_path}")
            return None

        features = {
            "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y=y)),
            "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            "rmse": np.mean(librosa.feature.rms(y=y)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_bandwidth": np.mean(
                librosa.feature.spectral_bandwidth(y=y, sr=sr)
            ),
            "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "spectral_contrast": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
            "tonnetz": np.mean(librosa.feature.tonnetz(y=y, sr=sr)),
            "spectral_flatness": np.mean(librosa.feature.spectral_flatness(y=y)),
            "mel_spectrogram": np.mean(librosa.feature.melspectrogram(y=y, sr=sr)),
            "chroma_cens": np.mean(librosa.feature.chroma_cens(y=y, sr=sr)),
        }

        # Tính tempo (beat per minute)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["beat_per_minute"] = float(tempo)

        # Trích xuất MFCC (64 giá trị)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        for i in range(64):
            features[f"mfcc_{i}"] = np.mean(log_mel_spec[i])

        # convert all values intp float32
        features = {key: float(value) for key, value in features.items()}

        return np.array(list(features.values()), dtype=np.float32)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        # x_sigmoid = torch.sigmoid(x)
        x_sigmoid = x  # without sigmoid since it has been computed
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()
