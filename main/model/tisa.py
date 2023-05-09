import pdb

import torch

from torch import nn


class Tisa(nn.Module):
    def __init__(self, num_attention_heads: int = 12, num_kernels: int = 5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kernels = num_kernels

        self.kernel_offsets = nn.Parameter(
            torch.Tensor(self.num_kernels, self.num_attention_heads)
        )
        self.kernel_amplitudes = nn.Parameter(
            torch.Tensor(self.num_kernels, self.num_attention_heads)
        )
        self.kernel_sharpness = nn.Parameter(
            torch.Tensor(self.num_kernels, self.num_attention_heads)
        )
        self._init_weights()

    def create_relative_offsets(self, seq_len: int):
        """Creates offsets for all the relative distances between
        -seq_len + 1 to seq_len - 1."""
        return torch.arange(-seq_len, seq_len + 1)

    def compute_positional_scores(self, relative_offsets):
        """Takes seq_len and outputs position scores for each relative distance.
        This implementation uses radial basis functions. Override this function to
        use other scoring functions than the example in the paper."""
        rbf_scores = (
            self.kernel_amplitudes.unsqueeze(-1)
            * torch.exp(
                -torch.abs(self.kernel_sharpness.unsqueeze(-1))
                * ((self.kernel_offsets.unsqueeze(-1) - relative_offsets) ** 2)
            )
        ).sum(axis=0)
        return rbf_scores

    def scores_to_toeplitz_matrix(self, positional_scores, seq_len: int):
        """Converts the TISA positional scores into the final matrix for the
        self-attention equation. PRs with memory and/or speed optimizations are
        welcome."""
        deformed_toeplitz = (
            (
                (torch.arange(0, -(seq_len ** 2), step=-1) + (seq_len - 1)).view(
                    seq_len, seq_len
                )
                + (seq_len + 1) * torch.arange(seq_len).view(-1, 1)
            )
            .view(-1)
            .long()
            .to(device=positional_scores.device)
        )
        expanded_positional_scores = torch.take_along_dim(
            positional_scores, deformed_toeplitz.view(1, -1), 1
        ).view(self.num_attention_heads, seq_len, seq_len)
        return expanded_positional_scores

    def forward(self, seq_len: int):
        """Computes the translation-invariant positional contribution to the
        attention matrix in the self-attention module of transformer models."""
        if not self.num_kernels:
            return torch.zeros((self.num_attention_heads, seq_len, seq_len))
        positional_scores_vector = self.compute_positional_scores(
            self.create_relative_offsets(seq_len)
        )
        positional_scores_matrix = self.scores_to_toeplitz_matrix(
            positional_scores_vector, seq_len
        )
        return positional_scores_matrix

    def visualize(self, seq_len: int = 10, attention_heads=None):
        """Visualizes the TISA interpretability by plotting position scores as
        a function of relative distance for each attention head."""
        if attention_heads is None:
            attention_heads = list(range(self.num_attention_heads))
        import matplotlib.pyplot as plt

        x = self.create_relative_offsets(seq_len).detach().numpy()
        y = (
            self.compute_positional_scores(self.create_relative_offsets(seq_len))
            .detach()
            .numpy()
        )
        for i in attention_heads:
            plt.plot(x, y[i])
        plt.savefig('./pic-tisa.png')
        plt.show()

    def _init_weights(self):
        """Initialize the weights"""
        ampl_init_mean = 0.1
        sharpness_init_mean = 0.1
        torch.nn.init.normal_(self.kernel_offsets, mean=0.0, std=5.0)
        torch.nn.init.normal_(
            self.kernel_amplitudes, mean=ampl_init_mean, std=0.1 * ampl_init_mean
        )
        torch.nn.init.normal_(
            self.kernel_sharpness,
            mean=sharpness_init_mean,
            std=0.1 * sharpness_init_mean,
        )


def main():
    tisa = Tisa()
    positional_scores = tisa(20)
    pdb.set_trace()
    tisa.visualize(seq_len=20)


if __name__ == "__main__":
    main()

