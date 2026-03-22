from __future__ import annotations

from dataclasses import dataclass


try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled by the training script
    raise RuntimeError(
        "PyTorch is required for the TabTransformer model. Install torch before importing this module."
    ) from exc


@dataclass(slots=True)
class TabTransformerConfig:
    num_features: int
    num_classes: int
    d_token: int = 64
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1
    mlp_hidden_dim: int = 128


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        self.column_embedding = nn.Parameter(torch.empty(num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.column_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) + self.column_embedding.unsqueeze(0)


class TabTransformerClassifier(nn.Module):
    def __init__(self, config: TabTransformerConfig) -> None:
        super().__init__()
        if config.d_token % config.n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")

        self.tokenizer = NumericalFeatureTokenizer(config.num_features, config.d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_token,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # Nested tensor fast-path is not used with norm_first=True; disable to avoid runtime warning noise.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_token),
            nn.Linear(config.d_token, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        encoded = self.transformer(torch.cat([cls_tokens, tokens], dim=1))
        pooled = encoded[:, 0, :]
        return self.head(pooled)
