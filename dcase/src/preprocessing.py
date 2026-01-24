import torch
import librosa
import os
import numpy as np


class MultiStreamPreprocessor:
    """Generate multiple audio representations for ensemble training."""

    def __init__(self, sample_rate: int = 44100, cache_dir: str = None):
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir

    def process(self, audio_stereo: torch.Tensor, cache_key: str = None) -> dict:
        """
        Args:
            audio_stereo: (2, TIME) stereo audio
            cache_key: unique identifier for caching hpspspsp
        Returns:
            dict of (1, TIME) tensors
        """
        device = audio_stereo.device

        L = audio_stereo[0:1]
        R = audio_stereo[1:2]
        mid = (L + R) / 2
        side = (L - R) / 2

        # try loading cached hspspps
        harmonic, percussive = self._get_hpss(mid, cache_key)

        return {
            'left': L,
            'right': R,
            'mid': mid,
            'side': side,
            'harmonic': harmonic.to(device),
            'percussive': percussive.to(device),
        }

    def _get_hpss(self, mid: torch.Tensor, cache_key: str):
        if self.cache_dir and cache_key:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}_hpss.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                return (
                    torch.from_numpy(data['harmonic'])[None, :].float(),
                    torch.from_numpy(data['percussive'])[None, :].float()
                )

        # compute hpss # why is this so hard to type hpspspsp meow
        mono_np = mid.squeeze().cpu().numpy()
        harmonic, percussive = librosa.effects.hpss(mono_np)

        if self.cache_dir and cache_key:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(cache_path, harmonic=harmonic, percussive=percussive)

        return (
            torch.from_numpy(harmonic)[None, :].float(),
            torch.from_numpy(percussive)[None, :].float()
        )
