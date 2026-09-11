import torch, torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, model_dim=128, max_len=1000):
        super().__init__()
        self.pe = nn.Embedding(max_len, model_dim)
    def forward(self, x):
        length = x.size(1)
        positions = torch.arange(0, length, device=x.device).unsqueeze(0)
        return x + self.pe(positions)

class PreLNTransformerLayer(nn.Module):
    def __init__(self, model_dim=128, nhead=8, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, model_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)
        return x

class TransformerWithPosEncoding(nn.Module):
    def __init__(self, model_dim=128, nhead=8, num_layers=8, dropout=0.2):
        super().__init__()
        self.pos_encoder = LearnedPositionalEncoding(model_dim)
        self.layers = nn.ModuleList([PreLNTransformerLayer(model_dim, nhead, dropout) for _ in range(num_layers)])
    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x

class EOG_NET(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=15),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.transformer = TransformerWithPosEncoding(model_dim=128, nhead=8, num_layers=4, dropout=0.2)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 1))
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        return self.classifier(x).squeeze(1)
