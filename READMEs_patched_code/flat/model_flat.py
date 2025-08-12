from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

def bucket_distance(dist, max_dist=20):
    dist = dist.clamp(min=-max_dist, max=max_dist)
    return dist + max_dist  # [0..2*max_dist]

class RelPosBias(nn.Module):
    def __init__(self, n_heads: int, max_dist: int = 20, type_bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        self.rel = nn.Embedding(2*max_dist + 1, n_heads)  # per-head scalar bias
        self.type_bias = type_bias
        if type_bias:
            # 2 types: 0=char, 1=word -> 4 pair types
            self.tt = nn.Parameter(torch.zeros(n_heads, 2, 2))  # [H, src_t, tgt_t]

    def forward(self, centers_i: torch.Tensor, centers_j: torch.Tensor, types_i: torch.Tensor, types_j: torch.Tensor):
        """
        centers_*: [B,N] (float or long)
        types_*:   [B,N] in {0,1}
        return: bias [B,H,N,N]
        """
        B, N = centers_i.size()
        ci = centers_i.unsqueeze(-1).expand(B, N, N)
        cj = centers_j.unsqueeze(-2).expand(B, N, N)
        dist = ci - cj  # [B,N,N]  可能是浮点（包含0.5）
        idx = bucket_distance(dist, self.max_dist)
        idx = torch.round(idx).to(torch.long)  # <- 关键：转成整数索引
        rel_bias = self.rel(idx)  # OK

        rel_bias = rel_bias.permute(0,3,1,2)  # [B,H,N,N]
        if self.type_bias:
            ti = types_i.unsqueeze(-1).expand(B, N, N)
            tj = types_j.unsqueeze(-2).expand(B, N, N)
            tb = self.tt[:, ti, tj]  # [H,B,N,N] with broadcasting
            tb = tb.permute(1,0,2,3) # [B,H,N,N]
            rel_bias = rel_bias + tb
        return rel_bias

class LatticeSelfAttnLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, d_ff=1024, dropout=0.1, max_dist=20):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.bias = RelPosBias(n_heads, max_dist=max_dist, type_bias=True)

    def forward(self, x, node_mask, centers, types):
        """
        x: [B,N,D], node_mask: [B,N] (1/0), centers: [B,N] (float), types: [B,N]
        """
        B,N,D = x.size()
        H = self.n_heads; Dh = self.d_head
        qkv = self.qkv(x)  # [B,N,3D]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        def split(t):
            return t.view(B, N, H, Dh).permute(0,2,1,3)  # [B,H,N,Dh]
        q = split(q); k = split(k); v = split(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh**0.5)  # [B,H,N,N]
        # add relative biases
        rb = self.bias(centers, centers, types, types)  # [B,H,N,N]
        scores = scores + rb
        # mask: invalid nodes -> -inf
        mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        scores = scores.masked_fill(mask==0, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        y = torch.matmul(att, v)  # [B,H,N,Dh]
        y = y.permute(0,2,1,3).contiguous().view(B,N,D)
        y = self.out_proj(y)
        x = self.norm1(x + self.drop(y))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

class FlatLatticeNER(BertPreTrainedModel):
    """
    扁平词格 Transformer（稳健简化版）：
    - 字节点用 BERT 首子词隐向量
    - 词节点 = span 内字隐向量平均 + 词嵌入（映射到 d_model 后相加）
    - 多层带相对位置偏置的自注意力（以 span 中心为“位置”）
    - 取字节点输出 -> 线性 -> CRF（在“字序列”上解码）
    """
    def __init__(self, config,
                 num_labels: int = None,
                 lexicon_size: int = None,
                 lex_dim: int = None,
                 n_layers: int = 2, n_heads: int = 8, d_ff: int = 1024,
                 max_dist: int = 20):
        super().__init__(config)
        if num_labels is None: num_labels = int(getattr(config, "num_labels", 2))
        if lexicon_size is None: lexicon_size = int(getattr(config, "lexicon_size", 0))
        if lex_dim is None: lex_dim = int(getattr(config, "lex_dim", 100))

        # 回写 config（保存时可复用）
        config.num_labels = num_labels
        config.lexicon_size = lexicon_size
        config.lex_dim = lex_dim
        config.n_layers = n_layers
        config.n_heads = n_heads
        config.d_ff = d_ff
        config.max_dist = max_dist

        self.num_labels = num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        D = config.hidden_size
        self.lex_emb = nn.Embedding(lexicon_size + 1, lex_dim, padding_idx=0)
        self.lex_proj = nn.Linear(lex_dim, D)

        self.layers = nn.ModuleList([
            LatticeSelfAttnLayer(d_model=D, n_heads=n_heads, d_ff=d_ff, dropout=getattr(config, "hidden_dropout_prob", 0.1), max_dist=max_dist)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(D, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        # 初始化更稳一些
        nn.init.normal_(self.lex_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.lex_proj.weight); nn.init.zeros_(self.lex_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)
        with torch.no_grad():
            self.crf.start_transitions.zero_()
            self.crf.end_transitions.zero_()
            self.crf.transitions.zero_()

        self.post_init()

    def forward(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        node_spans: torch.Tensor,     # [B,N,2] 字下标 (st,ed)
        node_types: torch.Tensor,     # [B,N]  0=char,1=word
        node_lex_ids: torch.Tensor,   # [B,N]
        node_mask: torch.Tensor,      # [B,N]  1/0
        char_token_pos: torch.Tensor, # [B,C]  每字首子词 token 下标
        char_mask: torch.Tensor,      # [B,C]  字有效位
        labels_char: Optional[torch.Tensor]=None,
        return_dict: bool=True
    ):
        B = input_ids.size(0)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, return_dict=True)
        seq = self.dropout(outputs.last_hidden_state)  # [B,L,D]
        D = seq.size(-1)

        # 取每个“字”的 BERT 隐向量（按首子词位置）
        C = char_token_pos.size(1)
        idx = char_token_pos.clamp(min=0)
        gather = torch.gather(seq, 1, idx.unsqueeze(-1).expand(B, C, D))  # [B,C,D]
        char_hidden = gather * (char_mask.unsqueeze(-1).float())

        # 初始节点向量：前 C 个是字节点，剩下是词节点
        N = node_spans.size(1)
        x = seq.new_zeros((B, N, D))

        # 字节点直接放入（节点顺序：前 C 个就是 0..C-1）
        x[:, :C, :] = char_hidden

        # 词节点：平均其 span 内的字隐向量 + 词嵌入
        if N > C:
            spans = node_spans[:, C:, :]  # [B,Nw,2]
            Nw = spans.size(1)
            word_repr = seq.new_zeros((B, Nw, D))
            for b in range(B):
                for i in range(Nw):
                    st = int(spans[b,i,0].item()); ed = int(spans[b,i,1].item())
                    if st<=ed and ed < C:
                        seg = char_hidden[b, st:ed+1, :]
                        if seg.numel() > 0:
                            word_repr[b,i,:] = seg.mean(dim=0)
            # 加上词嵌入映射
            lex_ids = node_lex_ids[:, C:]  # [B,Nw]
            lex_vec = self.lex_proj(self.lex_emb(lex_ids))  # [B,Nw,D]
            word_repr = word_repr + lex_vec
            x[:, C:, :] = word_repr

        # 自注意力编码（带相对位置信息）
        centers = ((node_spans[...,0] + node_spans[...,1]).float()) / 2.0  # [B,N]
        types = node_types
        for layer in self.layers:
            x = layer(x, node_mask=node_mask, centers=centers, types=types)

        # 取字节点输出 -> 分类 -> CRF（按字序列；长度可变，用 char_mask 遮罩）
        char_out = x[:, :C, :]  # [B,C,D]
        emissions = self.classifier(char_out)  # [B,C,num_labels]
        emissions = torch.nan_to_num(emissions, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-20, 20)

        loss = None
        if labels_char is not None:
            loss = -self.crf(emissions, labels_char, mask=char_mask.bool(), reduction='mean')
        preds = self.crf.decode(emissions, mask=char_mask.bool())

        if return_dict:
            return {"loss": loss, "logits": emissions, "pred_tags": preds, "mask": char_mask}
        else:
            return loss, emissions, preds, char_mask
