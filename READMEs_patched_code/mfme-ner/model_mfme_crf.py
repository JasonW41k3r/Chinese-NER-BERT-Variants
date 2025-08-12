# model_mfme_crf.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class MFMEForTokenClassification(BertPreTrainedModel):
    """
    多特征记忆编码（稳定版）：
    - lexicon: token对齐字符位置的K个候选词，安全softmax读出（无候选直接0向量）
    - bigram/pos/type: 字符级嵌入后gather到token位置
    - 融合：concat([bert_tok, mem]) -> 投影 -> CRF
    - 稳定性：安全softmax、emissions限幅/去NaN、稳定初始化、稳健CRF掩码
    - 关键：构造函数参数均可选，默认从 config 兜底，支持 from_pretrained(best_path) 直接加载
    """
    def __init__(self, config,
                 num_labels: int = None,
                 lexicon_size: int = None, bigram_size: int = None, pos_size: int = None, type_size: int = None,
                 lex_dim: int = None, bigram_dim: int = None, pos_dim: int = None, type_dim: int = None,
                 mem_hidden: int = None):
        super().__init__(config)

        # —— 从 config 兜底（确保 from_pretrained 不缺参）—— #
        if num_labels   is None: num_labels   = int(getattr(config, "num_labels", 2))
        if lexicon_size is None: lexicon_size = int(getattr(config, "lexicon_size", 0))
        if bigram_size  is None: bigram_size  = int(getattr(config, "bigram_size", 0))
        if pos_size     is None: pos_size     = int(getattr(config, "pos_size", 0))
        if type_size    is None: type_size    = int(getattr(config, "type_size", 6))
        if lex_dim      is None: lex_dim      = int(getattr(config, "lex_dim", 100))
        if bigram_dim   is None: bigram_dim   = int(getattr(config, "bigram_dim", 50))
        if pos_dim      is None: pos_dim      = int(getattr(config, "pos_dim", 16))
        if type_dim     is None: type_dim     = int(getattr(config, "type_dim", 8))
        if mem_hidden   is None: mem_hidden   = int(getattr(config, "mem_hidden", 128))

        # 同步回 config（方便保存/追踪）
        config.num_labels = num_labels
        config.lexicon_size = lexicon_size
        config.bigram_size = bigram_size
        config.pos_size = pos_size
        config.type_size = type_size
        config.lex_dim = lex_dim
        config.bigram_dim = bigram_dim
        config.pos_dim = pos_dim
        config.type_dim = type_dim
        config.mem_hidden = mem_hidden

        self.num_labels = num_labels

        # === Encoder ===
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        # === Embeddings ===
        self.lex_emb    = nn.Embedding(lexicon_size + 1, lex_dim, padding_idx=0)
        self.bigram_emb = nn.Embedding(bigram_size + 1, bigram_dim, padding_idx=0)
        self.pos_emb    = nn.Embedding(pos_size + 1, pos_dim, padding_idx=0)
        self.type_emb   = nn.Embedding(type_size + 1, type_dim, padding_idx=0)

        # lexicon attention
        self.q_proj = nn.Linear(config.hidden_size, lex_dim, bias=False)

        # memory proj
        mem_in = lex_dim + bigram_dim + pos_dim + type_dim
        self.mem_proj = nn.Sequential(
            nn.Linear(mem_in, mem_hidden),
            nn.GELU(),
            nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        )

        # fuse + classifier + CRF
        self.fuse = nn.Sequential(
            nn.Linear(config.hidden_size + mem_hidden, config.hidden_size),
            nn.GELU(),
            nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        )
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        # —— 稳定初始化 —— #
        nn.init.xavier_uniform_(self.q_proj.weight)
        for m in self.mem_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.fuse:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.normal_(self.lex_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bigram_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()
            self.crf.start_transitions.zero_()
            self.crf.end_transitions.zero_()
            self.crf.transitions.zero_()

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        token2char_index: Optional[torch.Tensor] = None,
        char_lex_ids: Optional[torch.Tensor] = None,      # [B,C,K]
        char_bigram_ids: Optional[torch.Tensor] = None,   # [B,C]
        char_pos_ids: Optional[torch.Tensor] = None,      # [B,C]
        char_type_ids: Optional[torch.Tensor] = None,     # [B,C]
        return_dict: bool = True,
    ) -> Tuple:
        # === BERT ===
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        seq_out = self.dropout(outputs.last_hidden_state)  # [B,L,H]
        B, L, H = seq_out.size()
        device = seq_out.device

        # ---- token->char gather ----
        if token2char_index is None:
            token2char_index = torch.full((B,L), -1, dtype=torch.long, device=device)
        idx = token2char_index.clamp(min=0)

        # 1) lexicon 注意力（安全 softmax）
        if char_lex_ids is not None:
            K = char_lex_ids.size(-1)
            Dlex = self.lex_emb.embedding_dim
            lex_emb = self.lex_emb(char_lex_ids)  # [B,C,K,D]
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, K, Dlex)
            lex_tok = torch.gather(lex_emb, 1, idx_exp)  # [B,L,K,D]

            k_ids = torch.gather(char_lex_ids, 1, idx.unsqueeze(-1).expand(B, L, K))
            k_mask = (k_ids > 0)  # [B,L,K]

            q = self.q_proj(seq_out).unsqueeze(-2)        # [B,L,1,D]
            scores = (q * lex_tok).sum(-1)                # [B,L,K]
            scores = scores.masked_fill(~k_mask, -1e9)
            att = torch.softmax(scores, dim=-1)           # [B,L,K]
            has_any = k_mask.any(dim=-1, keepdim=True)    # [B,L,1]
            att = att * has_any                           # 无候选 -> 全0
            lex_mem = (att.unsqueeze(-1) * lex_tok).sum(-2)  # [B,L,D]
            valid_tok = (token2char_index >= 0).unsqueeze(-1).float()
            lex_mem = lex_mem * valid_tok
        else:
            lex_mem = seq_out.new_zeros((B, L, self.lex_emb.embedding_dim))

        # 2) bigram / pos / type
        def gather_char_emb(char_ids, emb):
            D = emb.embedding_dim
            emb_ch = emb(char_ids)                          # [B,C,D]
            idx_exp = idx.unsqueeze(-1).expand(B, L, D)
            tok = torch.gather(emb_ch, 1, idx_exp)          # [B,L,D]
            valid_tok = (token2char_index >= 0).unsqueeze(-1).float()
            return tok * valid_tok

        bigram_tok = gather_char_emb(char_bigram_ids, self.bigram_emb) if char_bigram_ids is not None else seq_out.new_zeros((B,L,self.bigram_emb.embedding_dim))
        pos_tok    = gather_char_emb(char_pos_ids,    self.pos_emb)    if char_pos_ids is not None    else seq_out.new_zeros((B,L,self.pos_emb.embedding_dim))
        type_tok   = gather_char_emb(char_type_ids,   self.type_emb)   if char_type_ids is not None   else seq_out.new_zeros((B,L,self.type_emb.embedding_dim))

        # ---- 融合 + 分类 ----
        mem = torch.cat([lex_mem, bigram_tok, pos_tok, type_tok], dim=-1)   # [B,L,*]
        mem = self.mem_proj(mem)                                            # [B,L,M]
        fused = self.fuse(torch.cat([seq_out, mem], dim=-1))                # [B,L,H]
        emissions = self.classifier(fused)                                   # [B,L,C]

        # === 数值稳定：去 NaN/Inf + 限幅 ===
        emissions = torch.nan_to_num(emissions, nan=0.0, posinf=1e4, neginf=-1e4)
        emissions = emissions.clamp_(-20.0, 20.0)

        # === CRF 掩码：attention_mask ∧ 仅首子词；并确保第一步为 1 ===
        crf_mask = attention_mask.bool()
        if valid_mask is not None:
            first_col = torch.ones_like(valid_mask[:, :1], dtype=torch.bool)
            valid_positions = torch.cat([first_col, valid_mask[:, 1:].bool()], dim=1)
            crf_mask = crf_mask & valid_positions

        loss = None
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')

        pred_paths = self.crf.decode(emissions, mask=crf_mask)

        if return_dict:
            return {"loss": loss, "logits": emissions, "pred_tags": pred_paths, "mask": crf_mask}
        else:
            return loss, emissions, pred_paths, crf_mask
