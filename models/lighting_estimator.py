import torch
import torch.nn as nn
from torchvision import models


class LightingEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.out_dim = config.out_dim
        self.mlp_dim = config.mlp_dim
        self.dropout_rate = config.dropout_rate
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads

        base_model = models.resnet18(pretrained=False)
        self.resnet18 = nn.Sequential(*(list(base_model.children())[:-2]))

        encoder_layer = nn.TransformerEncoderLayer(
            self.out_dim,
            self.num_heads,
            self.mlp_dim,
            self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.linear = nn.Linear(self.out_dim, 3)
        self.linear2 = nn.Linear(self.out_dim, 9)

    def forward(self, inputs: torch.Tensor, pos_enc_2d: torch.Tensor) -> torch.Tensor:
        # inputs - (B, S, 3, 224, 224), pos_enc_2d - (B, S, 49, 512)
        b, s, c, h, w = inputs.shape

        # CNN feature extraction
        features = self.resnet18(inputs.view(b * s, c, h, w))
        _, f_c, f_h, f_w = features.shape
        features = features.reshape(b, s * f_h * f_w, f_c)

        # Reshape position encoding
        pos_enc_2d = pos_enc_2d.view(b, s * f_h * f_w, f_c)

        # Combine features with position encodings
        combined_features = features + pos_enc_2d
        enc_output = self.dropout1(self.encoder(combined_features))

        sunpos = self.linear(enc_output)
        params = self.linear2(enc_output)

        return sunpos, params


def get_module(cfg):
    lighting_estimator = LightingEstimator(cfg)

    if hasattr(cfg, "checkpoint"):
        lighting_estimator.load_state_dict(torch.load(cfg.checkpoint))

    return lighting_estimator
