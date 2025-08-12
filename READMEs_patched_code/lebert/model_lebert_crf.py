
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class LeBertCRF(BertPreTrainedModel):
    def __init__(self, config, lexicon_size: int, lex_dim: int = 100):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        self.lex_emb = nn.Embedding(lexicon_size + 1, lex_dim, padding_idx=0)
        self.fuse = nn.Linear(config.hidden_size + lex_dim, config.hidden_size)
        self.act = nn.GELU()

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        token2char_index: Optional[torch.Tensor] = None,
        char_lex_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Tuple:
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        seq_out = self.dropout(outputs.last_hidden_state)  # [B,L,H]

        # Build char-level lexicon embedding: char_lex_ids [B, C, K]
        # average embeddings per char over K (non-zero ids)
        if char_lex_ids is not None:
            emb = self.lex_emb(char_lex_ids)              # [B, C, K, D]
            mask = (char_lex_ids > 0).unsqueeze(-1).float()  # [B, C, K, 1]
            sum_emb = (emb * mask).sum(dim=2)             # [B, C, D]
            cnt = mask.sum(dim=2).clamp(min=1.0)          # [B, C, 1]
            char_lex = sum_emb / cnt                      # [B, C, D]
        else:
            B, L, H = seq_out.size()
            char_lex = seq_out.new_zeros((B, L, 100))     # fallback

        # Map char-level lex to token positions via token2char_index [B, L]
        if token2char_index is not None:
            B, L, _ = seq_out.size()
            D = char_lex.size(-1)
            idx = token2char_index.clamp(min=0)           # -1 -> 0
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D) # [B, L, D]
            # gather along char dimension (dim=1)
            # Need char_lex shaped [B, C, D]; idx along dim=1
            token_lex = torch.gather(char_lex, 1, idx_exp)  # [B, L, D]
            tok_mask = (token2char_index >= 0).unsqueeze(-1).float()
            token_lex = token_lex * tok_mask
        else:
            token_lex = seq_out.new_zeros((seq_out.size(0), seq_out.size(1), 100))

        # Fuse and classify
        fused = torch.cat([seq_out, token_lex], dim=-1)
        fused = self.act(self.fuse(fused))
        fused = self.dropout(fused)
        emissions = self.classifier(fused)

        # CRF mask: keep simple to avoid "first timestep must be on" issue
        crf_mask = attention_mask.bool()

        loss = None
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')

        pred_paths = self.crf.decode(emissions, mask=crf_mask)

        if return_dict:
            return {"loss": loss, "logits": emissions, "pred_tags": pred_paths, "mask": crf_mask}
        else:
            return loss, emissions, pred_paths, crf_mask
