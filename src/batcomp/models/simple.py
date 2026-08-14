from __future__ import annotations

import torch
import torch.nn.functional as F
from configmixin import ConfigMixin, register_to_config
from torch import nn
from x_transformers import Decoder


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SimpleModel(nn.Module, ConfigMixin):
    r"""Bare bone model with a state encoder and temporal decoder.

    Implements the toy model with a simple GLU local state encoder with
    residual connection and a decoder backbone to aggregate sequence of
    stete latents.
    """

    config_name = "model.json"

    @register_to_config
    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_input_channels: int = 1,
        context_size: int = 64,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_input_channels = num_input_channels

        self.pos_embedding = nn.Embedding(context_size, hidden_size)
        self.encoder = MLP(hidden_size, hidden_size * 2)
        self.decoder = Decoder(
            dim=hidden_size,
            depth=num_hidden_layers,
            heads=num_attention_heads,
            use_rmsnorm=True,
            pre_norm=True,
            ff_glue=True,
            ff_mult=2,
        )
        self.lm_head = nn.Linear(hidden_size, hidden_size)

    def get_state_latents(self, x):
        x = torch.atleast_2d(x)
        return x + self.encoder(x)

    def forward(self, x):
        z = self.get_state_latents(x)
        p = self.pos_embedding(torch.arange(z.size(1)).to(z.device, dtype=torch.long))
        h = self.decoder(z + p)
        return self.lm_head(h), z, h
