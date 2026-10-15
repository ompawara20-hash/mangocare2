"""AA-ENet architecture (EfficientNet-B0 + CBAM + Transformer encoder).

Vendored from the MIT-licensed `mango-disease-ai` 0.1.3 package
(AIUB R&D ICCA Research Group, https://github.com/aiub-rd-icca/mango-disease-ai)
so the checkpoint can be loaded without that package's bundled pipeline.
Architecture code is unmodified; see MODEL_CARD.md for provenance and the MIT notice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

NUM_CLASSES = 7


class CBAM(nn.Module):
    """Channel + Spatial attention (Woo et al., 2018)."""

    def __init__(self, ch, r=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch),
        )
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        sa = torch.cat(
            [x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1
        )
        sa = torch.sigmoid(self.conv(sa))
        return x * sa


# ── AA-ENet: Proposed lightweight hybrid ─────────────────────────────────────
class AAENet(nn.Module):
    """
    EfficientNet-B0 backbone  →  1×1 conv reduce  →  CBAM  →  Transformer
    encoder (1 layer) with [CLS] token  →  GAP + CLS concat  →  classifier.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        embed=192,
        use_cbam=True,
        use_transformer=True,
    ):
        super().__init__()
        # Backbone — pretrained=False at load time; weights come from .pt file
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, global_pool=""
        )
        feat_ch = self.backbone.num_features  # 1280

        self.reduce = nn.Sequential(
            nn.Conv2d(feat_ch, embed, 1, bias=False),
            nn.BatchNorm2d(embed),
            nn.ReLU(inplace=True),
        )

        self.use_cbam = use_cbam
        self.cbam = CBAM(embed) if use_cbam else nn.Identity()

        self.use_transformer = use_transformer
        if use_transformer:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed))
            self.pos = nn.Parameter(torch.zeros(1, 50, embed))  # 49 + cls
            enc = nn.TransformerEncoderLayer(
                d_model=embed,
                nhead=4,
                dim_feedforward=embed * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(enc, num_layers=1)

        fuse_dim = embed * 2 if use_transformer else embed
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(fuse_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # [B, 1280, 7, 7]
        x = self.reduce(x)  # [B, embed, 7, 7]
        x = self.cbam(x)
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, embed]
        if self.use_transformer:
            b = x.size(0)
            tok = x.flatten(2).transpose(1, 2)  # [B, 49, embed]
            cls = self.cls.expand(b, -1, -1)
            tok = torch.cat([cls, tok], dim=1) + self.pos[:, : tok.size(1) + 1]
            tok = self.transformer(tok)
            feat = torch.cat([gap, tok[:, 0]], dim=1)
        else:
            feat = gap
        return self.head(self.drop(feat))


# ── Model loader ─────────────────────────────────────────────────────────────
