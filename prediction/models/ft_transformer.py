"""
FT-Transformer (Feature Tokenizer Transformer)
Paper: "Revisiting Deep Learning Models for Tabular Data" — Gorishniy et al., 2021
Adapted for India Road Accident Risk Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Feature Tokenizer ─────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token    = d_token
        self.weight     = nn.Parameter(torch.empty(n_features, d_token))
        self.bias       = nn.Parameter(torch.empty(n_features, d_token))
        self.cls_token  = nn.Parameter(torch.empty(1, d_token))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls_token, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # tokens: (B, n_features, d_token)
        cls = self.cls_token.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)
        # output: (B, n_features+1, d_token)


# ─── Transformer Block ──────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ffn_factor: float = 4/3, dropout: float = 0.1):
        super().__init__()
        assert d_token % n_heads == 0
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        ffn_dim    = int(d_token * ffn_factor)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


# ─── FT-Transformer ────────────────────────────────────────────────────────────

class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        ffn_factor: float = 4/3,
        dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ):
        super().__init__()
        self.tokenizer  = FeatureTokenizer(n_features, d_token)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_token, n_heads, ffn_factor, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(residual_dropout),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)           # (B, n_features+1, d_token)
        tokens = self.transformer(tokens)    # (B, n_features+1, d_token)
        cls    = tokens[:, 0, :]             # (B, d_token)
        return self.head(cls)                # (B, n_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    def predict_risk_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x)[:, 1]


# ─── Model Factory ─────────────────────────────────────────────────────────────

def build_model(n_features: int, size: str = 'medium', device: str = 'cuda') -> FTTransformer:
    configs = {
        'small':  dict(d_token=128, n_layers=2, n_heads=8,  dropout=0.1),
        'medium': dict(d_token=192, n_layers=3, n_heads=8,  dropout=0.1),
        'large':  dict(d_token=256, n_layers=6, n_heads=16, dropout=0.2),
    }
    cfg   = configs[size]
    model = FTTransformer(n_features=n_features, n_classes=2, **cfg)
    model = model.to(device)
    print(f"FT-Transformer ({size}) | params: {sum(p.numel() for p in model.parameters()):,} | device: {device}")
    return model


# ─── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(24, size='medium', device=device)
    x      = torch.randn(32, 24).to(device)
    print(f"Input:  {x.shape}")
    print(f"Output: {model(x).shape}")
    print(f"Risk:   {model.predict_risk_score(x)[:3]}")
    print("OK ✓")