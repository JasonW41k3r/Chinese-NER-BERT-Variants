# model_bert_crf.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

class BertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
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
        return_dict: bool = True,
    ) -> Tuple:
        # valid_mask: [B, L]，只在“首子词位置”设为1，其余子词为0
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')

        pred_paths = self.crf.decode(emissions, mask=mask)

        if return_dict:
            return {"loss": loss, "logits": emissions, "pred_tags": pred_paths, "mask": mask}
        else:
            return loss, emissions, pred_paths, mask
