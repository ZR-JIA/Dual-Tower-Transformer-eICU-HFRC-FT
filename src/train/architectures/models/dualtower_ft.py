"""
DualTower with an FT-Transformer right tower (DualTower-FT)

Architecture:
    Left tower:  one embedding per categorical feature (in the HF-CS study:
                 sex, ethnicity and hospital), as in DualTower
    Right tower: FTRightTower
                   NumericalFeatureTokenizer
                   -> prepend a learnable [CLS] token
                   -> TransformerEncoder (feed-forward width 4 x d_model, as in
                      FT-Transformer)
                   -> [CLS] output as the summary of the numerical features
    Fusion:      concatenation of both towers and an MLP head, as in DualTower

Differences from dualtower.py:
    - The right tower aggregates with a learnable [CLS] token instead of mean
      pooling, so the [CLS] token attends over all numerical feature tokens.
    - The feed-forward width is 4 x d_model (FT-Transformer default) instead
      of 2 x d_model.

Reference:
    Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data.
    NeurIPS 2021. https://arxiv.org/abs/2106.11959
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List

from .base import NumericalFeatureTokenizer

logger = logging.getLogger(__name__)


class FTRightTower(nn.Module):
    """
    FT-Transformer style right tower for numerical features.

    Replaces mean pooling with a learnable [CLS] token.
    The [CLS] token is prepended to the feature token sequence and
    processed jointly by the Transformer encoder. Its output serves
    as the aggregated representation of all numerical features.

    Pipeline:
        x [B, N]
        -> NumericalFeatureTokenizer -> tokens [B, N, d_model]
        -> prepend cls_token         -> tokens [B, N+1, d_model]
        -> TransformerEncoder        -> tokens [B, N+1, d_model]
        -> tokens[:, 0, :]           -> cls_out [B, d_model]

    pooling="mean" is an ablation switch: no [CLS] token is prepended and the
    encoder outputs are averaged; everything else stays the same.
    """

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float, pooling: str = "cls"):
        """
        Args:
            input_dim:  Number of numerical features
            d_model:    Token embedding dimension
            nhead:      Number of self-attention heads
            num_layers: Number of TransformerEncoder layers
            dropout:    Dropout rate
            pooling:    "cls" (default) or "mean" (ablation)
        """
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.tokenizer = NumericalFeatureTokenizer(input_dim, d_model)

        # Learnable [CLS] token, initialised to zero and learned in training
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,   # FT-Transformer convention (4x, not 2x)
            dropout=dropout,
            batch_first=True,
            norm_first=True                # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N_num]
        Returns:
            cls_out: [B, d_model]
        """
        B = x.size(0)
        tokens = self.tokenizer(x)                          # [B, N, d_model]
        if self.pooling == "mean":
            return self.transformer_encoder(tokens).mean(dim=1)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)            # [B, N+1, d_model]

        # Self-attention over all tokens (CLS + features)
        tokens = self.transformer_encoder(tokens)           # [B, N+1, d_model]

        # Return [CLS] output as the aggregated representation
        return tokens[:, 0, :]                              # [B, d_model]


class DualTowerFT(nn.Module):
    """
    DualTower with FT-Transformer Right Tower (DT-FT).

    Left tower:  learnable embeddings for the categorical features (unchanged).
    Right tower: FTRightTower, [CLS] token aggregation instead of mean pooling.
    Fusion:      concatenation, MLP, one logit (unchanged).

    Config key: model: dualtower_ft
    """

    def __init__(self, config: Dict[str, Any], input_dim: int):
        """
        Args:
            config:    Model configuration dict (expects 'architecture' key)
            input_dim: Total input feature dimension
        """
        super().__init__()
        self.config = config

        arch = config.get('architecture', {})
        if not arch:
            arch = config.get('model_config', {}).get('architecture', {})

        # Left tower: categorical embeddings (same as DualTower)
        self.cat_features_config = arch['cat_features']

        if 'num_indices' in arch:
            self.num_indices = arch['num_indices']
        else:
            cat_idx_set = set(f['index'] for f in self.cat_features_config)
            self.num_indices = [
                i for i in range(input_dim) if i not in cat_idx_set
            ]

        self.embeddings = nn.ModuleList()
        self.cat_indices = []
        total_emb_dim = 0
        for feat in self.cat_features_config:
            self.embeddings.append(
                nn.Embedding(feat['num_classes'], feat['emb_dim'])
            )
            self.cat_indices.append(feat['index'])
            total_emb_dim += feat['emb_dim']

        # Right tower: FT-Transformer with a [CLS] token
        d_model = arch['trans_d_model']
        num_feat_count = len(self.num_indices)

        self.right_tower = FTRightTower(
            input_dim=num_feat_count,
            d_model=d_model,
            nhead=arch.get('trans_nhead', 4),
            num_layers=arch.get('trans_num_layers', 2),
            dropout=arch['dropout'],
            pooling=arch.get('pooling', 'cls')
        )
        right_output_dim = d_model

        # Fusion layer (same as DualTower)
        fusion_in = total_emb_dim + right_output_dim
        fusion_hidden = arch['fusion_hidden_dim']

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(arch['dropout']),
            nn.Linear(fusion_hidden, 1)
        )

        # Runtime safeguard switch. When True (default), _compute_towers
        # replaces non-finite values, clamps categorical codes to their
        # vocabulary and clamps numerical values to +/-10. The ablation without
        # the safeguard and run_safeguard_stress_test.py set it to False.
        self.safeguard_enabled: bool = True

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualTowerFT: Initialized with FT-Transformer Right Tower "
            f"(CLS aggregation, FFN 4x) | {n_params:,} parameters | "
            f"safeguard_enabled={self.safeguard_enabled}"
        )

    def _compute_towers(self, x: torch.Tensor):
        """
        Compute the left and right tower outputs.

        Applies the runtime safeguard in three stages (non-finite guard,
        categorical clamp, numerical clamp) while ``self.safeguard_enabled``
        is True, which is the default. Only the ablation and the stress test
        switch it off.
        """
        safeguard = getattr(self, 'safeguard_enabled', True)

        # Stage A: replace NaN and +/-Inf with 0, the training mean after
        # scaling. On clean inputs nothing changes; otherwise a missed NaN
        # cannot spread through the network.
        if safeguard:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Left tower (categorical)
        emb_outs = []
        for idx, emb, feat in zip(
            self.cat_indices, self.embeddings, self.cat_features_config
        ):
            if safeguard:
                # Stage B: categorical clamp to embedding vocabulary range
                col_input = torch.clamp(
                    x[:, idx], min=0, max=feat['num_classes'] - 1
                ).long()
            else:
                # Without the safeguard: direct cast. Raises IndexError if a
                # code is outside the embedding's vocabulary.
                col_input = x[:, idx].long()
            emb_outs.append(emb(col_input))
        left_out = torch.cat(emb_outs, dim=1)                # [B, total_emb_dim]

        # Right tower (numerical, [CLS] aggregation)
        num_data = x[:, self.num_indices]
        if safeguard:
            # Stage C: clamp numerical values to +/-10 standard deviations
            # after scaling. Plausible values lie well inside +/-5, so the
            # clamp only bounds device errors and unit mistakes.
            num_data = torch.clamp(num_data, min=-10.0, max=10.0)
        right_out = self.right_tower(num_data)               # [B, d_model]

        return left_out, right_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            logits: [B]
        """
        left_out, right_out = self._compute_towers(x)

        # --- Fusion ---
        combined = torch.cat([left_out, right_out], dim=1)  # [B, total_emb_dim + d_model]
        logits = self.fusion(combined)                       # [B, 1]

        return logits.squeeze(-1)
