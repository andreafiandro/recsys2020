import pdb

import torch.nn as nn
from transformers import BertConfig, BertModel


class BERT(nn.Module):
    """Class wrapping BERT for sequence classification
    """
    def __init__(self, pretrained, n_labels, dropout_prob = .5, freeze_bert=None):
        super(BERT, self).__init__()
        self.config = BertConfig(_num_labels=n_labels)
        self.bert = BertModel(self.config).from_pretrained(pretrained, output_hidden_states=False, output_attentions=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(in_features=self.config.hidden_size, out_features=self.config._num_labels)
        
        if freeze_bert:
            self.freeze_layers(self.bert)


    def freeze_layers(self, layers):
        for param in layers.parameters():
            param.requires_grad = False     


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )

        cls_output = outputs[1]
        pooled_output = self.dropout(cls_output)
        logits = self.classifier(pooled_output)

        return logits, cls_output