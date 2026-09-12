"""Positional encoding + the NeRF MLP.

Vendored/adapted from yenchenlin/nerf-pytorch's run_nerf_helpers.py (MIT
license, see src/nerf/THIRD_PARTY_LICENSE), commit
63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd. Math is unchanged from the
original; only reorganized into this project's module layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder:
    """Positional encoding (NeRF paper, section 5.1)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        n_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=n_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires: int, use_encoding: bool = True):
    """Build a positional-encoding function.

    `multires` is the number of frequency bands (config field
    model.pos_encoding_freqs / model.dir_encoding_freqs). `use_encoding`
    replaces the original's `i_embed == -1` "no encoding" escape hatch.
    """
    if not use_encoding:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)  # noqa: E731
    return embed, embedder_obj.out_dim


class NeRF(nn.Module):
    """The NeRF MLP: 8 layers, 256 channels, one skip connection at layer 4."""

    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 3,
        input_ch_views: int = 3,
        output_ch: int = 4,
        skips=(4,),
        use_viewdirs: bool = False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        # Matches the official TF release's view-direction branch (a single
        # layer), not the deeper branch sketched in the paper's diagram —
        # see the original repo's comment on this exact discrepancy.
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, _ in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
