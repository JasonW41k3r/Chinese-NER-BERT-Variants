# model_spankl.py
from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class SpanKLForNER(BertPreTrainedModel):
    def __init__(self, config, num_labels: int = None, max_span_len: int = None, width_dim: int = 32):
        super().__init__(config)
        # ---- 从 config 兜底读取 ----
        if num_labels is None:
            num_labels = int(getattr(config, "num_labels", 0))
        if max_span_len is None:
            max_span_len = int(getattr(config, "max_span_len", 8))

        self.num_labels = num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        self.width_embeddings = nn.Embedding(max_span_len+1, width_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size*2 + width_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1)),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        self.max_span_len = max_span_len
        self.post_init()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                span_positions: torch.Tensor,   # [T,2]
                span_batch_idx: torch.Tensor,   # [T]
                span_labels: torch.Tensor = None,
                ) -> Dict[str, Any]:

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
        seq = self.dropout(outputs.last_hidden_state)  # [B,L,H]
        T = span_positions.size(0)
        if T == 0:
            logits = seq.new_zeros((0, self.num_labels))
            loss = logits.sum()*0
            return {"loss": loss, "logits": logits}

        h_start = seq[span_batch_idx, span_positions[:,0]]   # [T,H]
        h_end   = seq[span_batch_idx, span_positions[:,1]]   # [T,H]
        width = (span_positions[:,1] - span_positions[:,0] + 1).clamp(min=1, max=self.max_span_len)
        w_embed = self.width_embeddings(width)                # [T,Dw]
        rep = torch.cat([h_start, h_end, w_embed], dim=-1)    # [T,2H+Dw]
        logits = self.classifier(rep)                         # [T,C]

        out = {"logits": logits}
        if span_labels is not None:
            loss = nn.functional.cross_entropy(logits, span_labels)
            out["loss"] = loss
        return out
