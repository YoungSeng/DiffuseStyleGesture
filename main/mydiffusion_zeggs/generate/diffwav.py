import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import sys
[sys.path.append(i) for i in ['.', '..', '../../..']]
from generate.generate import WavEncoder

from mydiffwave.src.diffwave.model import DiffWave
from mydiffwave.src.diffwave.params import params
import numpy as np


class diffwav_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.WavEncoder = WavEncoder()
        self.diffwav_model = DiffWave(params)
        self.criterion = nn.SmoothL1Loss()

    def sample(self, batch_size, tmp_audio, beta, T):
        wav_feature = self.WavEncoder(tmp_audio).transpose(1, 2)  # (b, 240, 32)
        noisy_pose = torch.randn(batch_size, 240, 135).transpose(1, 2).to(tmp_audio.device)
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            noisy_pose = c1 * (noisy_pose - c2 * self.diffwav_model(noisy_pose, torch.tensor([T[n]], device=noisy_pose.device), wav_feature).squeeze(1))
            if n > 0:
                noise = torch.randn_like(noisy_pose)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                noisy_pose += sigma * noise
            noisy_pose = torch.clamp(noisy_pose, -1.0, 1.0)
        return noisy_pose.transpose(1, 2)

    def forward(self, noisy_pose, t, audio, noise):      # (b, len, 13)
        wav_feature = self.WavEncoder(audio).transpose(1, 2)        # (b, 240, 32)
        predicted = self.diffwav_model(noisy_pose, t, wav_feature)
        loss = self.criterion(predicted, noise)
        return loss


if __name__ == '__main__':
    '''
    cd mydiffusion/generate/
    python diffwav.py
    '''
    # z = torch.arange(0, 60).reshape(2, 30)

    device = torch.device('cuda:2')
    audio = torch.rand(2, 64000).to(device)
    pose = torch.rand(2, 240, 135).transpose(1, 2).to(device)
    model = diffwav_model().to(device)

    n_frames = 240
    n_pose_dims = 135
    n_audio_dim = 32
    hop_samples = 1

    N = pose.shape[0]  # 1, 15872
    device = pose.device

    beta = np.linspace(1e-4, 0.05, 50)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32)).to(device)

    t = torch.randint(0, len(beta), [N], device=pose.device)  # (batch)
    noise_scale = noise_level[t].unsqueeze(1).unsqueeze(1)  # (batch, 1)
    noise_scale_sqrt = noise_scale ** 0.5  # (batch, 1)
    noise = torch.randn_like(pose)  # (batch, 15872)
    noisy_pose = noise_scale_sqrt * pose + (1.0 - noise_scale) ** 0.5 * noise  # (batch, 15872)

    loss = model(noisy_pose, t, audio, pose)  # (batch, 1, 15872)
    print(loss)

    talpha = 1 - beta
    talpha_cum = np.cumprod(talpha)
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(beta)):
      for t in range(len(beta) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)

    tmp_audio = torch.rand(1, 64000).to(device)
    sampled_seq = model.sample(batch_size=1, tmp_audio=tmp_audio, beta=beta, T=T)
    print(sampled_seq.shape)      # (4, 32, 128)
