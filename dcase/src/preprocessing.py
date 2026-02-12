import torch
import os
import numpy as np
import librosa


class MultiStreamPreprocessor:
    """Generate multiple audio representations for ensemble training."""

    def __init__(
            self,
            sample_rate: int = 44100,
            cache_dir: str = None,
            use_fast_hpss: bool = False,
            hpss_kernel_size: int = 1024,
            hpss_stride: int = 256,
    ):
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.use_fast_hpss = use_fast_hpss
        self.hpss_kernel_size = hpss_kernel_size
        self.hpss_stride = hpss_stride

    def process(self, audio_stereo: torch.Tensor, cache_key: str = None) -> dict:
        """
        Args:
            audio_stereo: (2, TIME) stereo audio
            cache_key: unique identifier for caching hpspspsp
        Returns:
            dict of (1, TIME) tensors
        """
        device = audio_stereo.device
        if self.cache_dir and cache_key:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}_streams.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location=device, weights_only=True)
                return {k: v.to(device) for k, v in cached.items()}

        L = audio_stereo[0:1]
        R = audio_stereo[1:2]
        mid = (L + R) / 2
        side = (L - R) / 2

        if self.use_fast_hpss:
            harmonic, percussive = self._fast_hpss(
                mid,
                kernel_size=self.hpss_kernel_size,
                stride=self.hpss_stride,
            )
        else:
            harmonic, percussive = self._get_hpss(mid, cache_key)

        # bg = self._moving_average_fast(mid, win_size=int(2 * self.sample_rate))
        # fg = mid - bg

        streams = {
            'left': L, 'right': R, 'mid': mid, 'side': side,
            'harmonic': harmonic.to(device), 'percussive': percussive.to(device)
        }
        if self.cache_dir and cache_key:
            os.makedirs(self.cache_dir, exist_ok=True)
            try:
                torch.save({k: v.detach().cpu() for k, v in streams.items()}, cache_path)
            except Exception:
                pass
        return streams

    @staticmethod
    def _fast_hpss(y: torch.Tensor, kernel_size=1024, stride=256):
        # y: (1, T)
        T = y.shape[-1]
        pooled = torch.nn.functional.avg_pool1d(
            y.unsqueeze(0), kernel_size, stride=stride, padding=0
        ).squeeze(0)
        harmonic = torch.nn.functional.interpolate(
            pooled.unsqueeze(0), size=T, mode='linear', align_corners=False
        ).squeeze(0)
        percussive = y - harmonic
        return harmonic, percussive

    def _get_hpss(self, mid: torch.Tensor, cache_key: str):
        if self.cache_dir and cache_key:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}_hpss.npz")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                return (
                    torch.from_numpy(data['harmonic'])[None, :].float(),
                    torch.from_numpy(data['percussive'])[None, :].float()
                )

        # compute hpspps with Librosa (slow)
        mono_np = mid.squeeze().cpu().numpy()
        harmonic, percussive = librosa.effects.hpss(mono_np)

        if self.cache_dir and cache_key:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(cache_path, harmonic=harmonic, percussive=percussive)

        return (
            torch.from_numpy(harmonic)[None, :].float(),
            torch.from_numpy(percussive)[None, :].float()
        )

    @staticmethod
    def _moving_average_fast(x: torch.Tensor, win_size: int):
        # x: (1, T)
        T = x.shape[-1]
        x_unsq = x.unsqueeze(0)  # (1, 1, T)
        zeros = torch.zeros(1, 1, 1, device=x.device)
        cumsum = torch.cumsum(torch.cat([zeros, x_unsq], dim=-1), dim=-1)
        bg = (cumsum[..., win_size:] - cumsum[..., :-win_size]) / win_size
        pad = win_size // 2
        bg = torch.nn.functional.pad(bg, (pad, T - bg.shape[-1] - pad))
        return bg.squeeze(0)