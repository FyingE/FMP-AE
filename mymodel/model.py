import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # output shape: (batch, feature_dim, 1)
        )

    def forward(self, x):  # x: (batch, window_len, dim) => need to permute
        x = x.permute(0, 2, 1)  # to (batch, dim, window_len)
        x = self.conv_block(x)  # output: (batch, feature_dim, 1)
        return x.squeeze(-1)  # (batch, feature_dim)


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead):
        super(TransformerAutoencoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        src = src.permute(1, 0, 2)  # to (seq_len, batch, input_dim)
        memory = self.encoder(src)
        out = self.decoder(src, memory)
        out = self.output_proj(out)
        return out.permute(1, 0, 2)  # back to (batch, seq_len, input_dim)


def compute_inter_window_mp(features):
    # features: (batch, feature_dim)
    norms = torch.cdist(features, features, p=2)
    mask = torch.eye(norms.shape[0], device=features.device).bool()
    norms.masked_fill_(mask, float("inf"))
    return torch.min(norms, dim=1)[0]  # (batch,)


def compute_intra_window_mp(sub_features):
    # sub_features: (batch, num_segments, feature_dim)
    sim = F.cosine_similarity(
        sub_features.unsqueeze(2), sub_features.unsqueeze(1), dim=-1
    )
    eye = torch.eye(sim.shape[1], device=sim.device).bool().unsqueeze(0)
    sim.masked_fill_(eye, 0)
    return torch.sum(1 - sim, dim=(1, 2)) / (sim.shape[1] * (sim.shape[1] - 1))


class FMP_TAE(nn.Module):
    def __init__(
        self, input_dim, cnn_feature_dim, trans_hidden_dim, trans_layers, trans_heads
    ):
        super(FMP_TAE, self).__init__()
        self.cnn = CNNFeatureExtractor(input_dim, cnn_feature_dim)
        self.transformer = TransformerAutoencoder(
            input_dim=input_dim,
            hidden_dim=trans_hidden_dim,
            num_layers=trans_layers,
            nhead=trans_heads,
        )

    def forward(self, x):
        # x: (batch, window_len, input_dim)
        cnn_feat = self.cnn(x)  # (batch, feature_dim)
        recon_x = self.transformer(x)  # (batch, window_len, input_dim)
        return cnn_feat, recon_x

    def compute_anomaly_score(
        self, x, sub_segments, lambda1=1.0, lambda2=1.0, lambda3=1.0
    ):
        cnn_feat, recon_x = self.forward(x)
        rec_err = torch.mean((x - recon_x) ** 2, dim=(1, 2))

        inter_mp = compute_inter_window_mp(cnn_feat)
        intra_mp = compute_intra_window_mp(sub_segments)

        inter_w = F.softmax(inter_mp, dim=0)
        intra_w = F.softmax(intra_mp, dim=0)

        score = (
            lambda1 * torch.sum(inter_w * inter_mp)
            + lambda2 * torch.sum(intra_w * intra_mp)
            + lambda3 * torch.mean(rec_err)
        )
        return score, rec_err, inter_mp, intra_mp
