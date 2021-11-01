import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool1d


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MicroBlock(nn.Module):
    def __init__(self, nh, kernel_size):
        super().__init__()
        self.conv1 = ConvBNACT(nh, nh, kernel_size, groups=nh, padding=1)
        self.conv2 = ConvBNACT(nh, nh, 1)

    def forward(self, x):
        x = x + self.conv1(x)
        x = self.conv2(x)
        return x


class MicroNetV2(nn.Module):
    """
    attention model
    """
    def __init__(self, nh=64, depth=2, nclass=60, img_height=32):
        super().__init__()
        assert(nh >= 2)
        self.conv = ConvBNACT(3, nh, 4, 4)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(MicroBlock(nh, 3))

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        # LSTM Encoder
        linear_in = nh * int((img_height-(4-1)-1)/4 + 1)
        hidden = 64 if nh < 256 else nh//4
        self.rnn_encoder = nn.GRU(linear_in, hidden, batch_first=True)
        # LSTM Decoder
        self.rnn_decoder = nn.GRU(hidden, hidden, batch_first=True)
        # Decoder input embedding
        self.embedding = nn.Embedding(
            nclass, hidden, padding_idx=1)
        # attention layer
        self.conv1x1_1 = nn.Linear(hidden, nh)
        self.conv3x3_1 = nn.Conv2d(
            nh, nh, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(nh, 1)
        # Prediction layer
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(nh, nclass)

    def _attention(self, decoder_input, feat):
        y = self.rnn_decoder(decoder_input)[0] # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.size()
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat) # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1) # bsz * 1 * attn_size * h * w  

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1)) # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous() # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight) # bsz * (seq_len + 1) * h * w * 1
        
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()
        attn_feat = torch.sum(
            torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)

        return attn_feat

    def forward(self, x, targets):
        x_shape = x.size()
        x = self.conv(x)
        for block in self.blocks:
            feat = block(x)
        feat_v = self.flatten(feat)
        feat_v = adaptive_avg_pool1d(feat_v, int(x_shape[3]/4))
        feat_v = feat_v.permute(0, 2, 1)
        holistic_feat = self.rnn_encoder(feat_v)[0]
        out_enc = holistic_feat[:, -1, :]  # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)  # bsz * 1 * emb_dim
        tgt_embedding = self.embedding(targets) # bsz * (seq_len + 1) * C
        in_dec = torch.cat((out_enc, tgt_embedding), dim=1)
        attn_feat = self._attention(
            in_dec, feat)
        y = self.dropout(attn_feat)
        y = self.fc(y)
        return y


if __name__ == '__main__':
    import time
    x = torch.randn(1, 3, 32, 256)
    y = torch.arange(0, 12).view(1, -1).to(torch.long)
    model = MicroNetV2(32, depth=2, nclass=62, img_height=32)
    t0 = time.time()
    out = model(x, y)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)
    torch.save(model, 'test.pth')
    # from torchsummaryX import summary
    # summary(model, x)
