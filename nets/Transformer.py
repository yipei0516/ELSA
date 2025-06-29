import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import pdb

class HybridAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(HybridAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, video, audio):
        video_sa = self.layer(video, video, video)
        audio_sa = self.layer(audio, audio, audio)
        video_cma = self.layer(video_sa, audio_sa, audio_sa)
        audio_cma = self.layer(audio_sa, video_sa, video_sa)
        return video_cma, audio_cma


class SelfAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(SelfAttentionBlock, self).__init__()
        self.layer = attention_layer # MultiHeadAttention
        self.size = attention_layer.size

    def forward(self, feature):
        feature_sa = self.layer(feature, feature, feature)
        return feature_sa


class CrossAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(CrossAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, video, audio):
        video_cma = self.layer(video, audio, audio)
        audio_cma = self.layer(audio, video, video)
        return video_cma, audio_cma


class IntegrateAttentionBlock(nn.Module):
    def __init__(self, vselfattention_layer, aselfattention_layer, crossattention_layer, num_inputs):
        super(IntegrateAttentionBlock, self).__init__()
        self.vsablock = SelfAttentionBlock(vselfattention_layer) # vselfattention_layer: TransformerLayer
        self.asablock = SelfAttentionBlock(aselfattention_layer)
        self.cmablock = CrossAttentionBlock(crossattention_layer)
        self.lv1 = nn.Linear(num_inputs*2, num_inputs)
        self.lv2 = nn.Linear(num_inputs*2, num_inputs)
        self.la1 = nn.Linear(num_inputs*2, num_inputs)
        self.la2 = nn.Linear(num_inputs*2, num_inputs)

    def forward(self, video, audio):
        v_sa = self.vsablock(video)
        a_sa = self.asablock(audio)
        v_cma, a_cma = self.cmablock(video, audio)
        v_sq = torch.cat((v_sa, v_cma), -1)
        v_e1 = torch.sigmoid(self.lv1(v_sq))
        v_e2 = torch.sigmoid(self.lv2(v_sq))
        v_out = torch.mul(v_e1, v_sa) + torch.mul(v_e2, v_cma)
        a_sq = torch.cat((a_sa, a_cma), -1)
        a_e1 = torch.sigmoid(self.la1(a_sq))
        a_e2 = torch.sigmoid(self.la2(a_sq))
        a_out = torch.mul(a_e1, a_sa) + torch.mul(a_e2, a_cma)
        # v_out = self.lv(torch.cat((v_sa, v_cma), -1))
        # a_out = self.la(torch.cat((a_sa, a_cma), -1))
        return v_out, a_out


class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn # MultiHeadAttention
        self.feed_forward = feed_forward # PositionwiseFeedForward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # LayerNorm+Dropout
        self.size = size

    def forward(self, q, k, v):
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0]) # Self-Attention 子層 + 殘差連接 + LayerNorm
        return self.sublayer[1](q, self.feed_forward) # Feed-Forward Network 子層 + 殘差連接 + LayerNorm


class SublayerConnection(nn.Module): # Transformer 論文中每個子層（sublayer）所使用的殘差結構 + Layer Normalization（子層連接模組）
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout) # Dropout 防止 overfitting

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, masksize, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # [B, h, 10, 10]
    if masksize != 1: # local attention mask
        # 每個 query（第 i 個時間段）只關注附近的 masksize 範圍內的 key 值，忽略其他遠處時間段，模擬一種「局部注意力」。
        masksize = int(masksize / 2)
        mask = torch.ones(scores.size()).cuda()
        for i in range(mask.shape[2]):
            if i - masksize > 0:
                mask[:, :, i, :i - masksize] = 0
            if i + masksize + 1 < mask.shape[3]:
                mask[:, :, i, masksize + i + 1:] = 0
        # print(mask[0][0])
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def vanillattention(query, key, value, masksize=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # [B, h, 25, 10]
    # 沒有local attention mask，是一種全局注意力機制
    p_attn = F.softmax(scores, dim=-1) # [B, h, 25, 10]
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), p_attn
    return torch.matmul(p_attn, value), scores #! scores without softmax


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))] # [b, 10, h, d_k] -> [b, h, 10, d_k]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, self.masksize, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return out, self.attn


class PositionwiseFeedForward(nn.Module): # 它出現在每個 Encoder 和 Decoder layer 的 Attention 後面
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 第1層: 擴大維度 d_model → d_ff
        self.w_2 = nn.Linear(d_ff, d_model) # 第2層: 壓縮維度 d_ff → d_model
        self.dropout = nn.Dropout(dropout)  # dropout 防止過擬合

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output



class VanillaMultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        super(VanillaMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))] # [b, 10, h, d_k] -> [b, h, 10, d_k]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = vanillattention(query, key, value, self.masksize, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return out, self.attn


class VanillaTransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(VanillaTransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, q, k, v):
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0]) # Self-Attention 子層 + 殘差連接 + LayerNorm
        attn = self.self_attn(q, k, v)[1]
        return self.sublayer[1](q, self.feed_forward), attn


if __name__ == "__main__":

    query = torch.randn(2, 10, 512)
    key = torch.randn(2, 10, 512)
    value = torch.randn(2, 10, 512)
    nbatches = 2
    h = 8
    d_k = 64
    d_model = 512
    linears  = clones(nn.Linear(d_model, d_model), 4)
    query, key, value = [l(x).view(nbatches, -1, h, d_k).transpose(1, 2) for l, x in
                        zip(linears, (query, key, value))] # [b, 10, h, d_k] -> [b, h, 10, d_k]
    pdb.set_trace()