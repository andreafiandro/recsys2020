import pdb

import torch.nn as nn
from transformers import BertConfig, BertModel


class BERT(nn.Module):
    """Class wrapping BERT for sequence classification
    """
    def __init__(self, pretrained, n_labels, dropout_prob = .5, freeze_bert=None):
        """To change more parameters just set them in BertConfig()
            Possible parameters:
            BertConfig {
                "_num_labels": 2,
                "architectures": null,
                "attention_probs_dropout_prob": 0.1,
                "bad_words_ids": null,
                "bos_token_id": null,
                "decoder_start_token_id": null,
                "do_sample": false,
                "early_stopping": false,
                "eos_token_id": null,
                "finetuning_task": null,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "id2label": {
                    "0": "LABEL_0",
                    "1": "LABEL_1"
                },
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "is_decoder": false,
                "is_encoder_decoder": false,
                "label2id": {
                    "LABEL_0": 0,
                    "LABEL_1": 1
                },
                "layer_norm_eps": 1e-12,
                "length_penalty": 1.0,
                "max_length": 20,
                "max_position_embeddings": 512,
                "min_length": 0,
                "model_type": "bert",
                "n_labels": 2,
                "no_repeat_ngram_size": 0,
                "num_attention_heads": 12,
                "num_beams": 1,
                "num_hidden_layers": 12,
                "num_return_sequences": 1,
                "output_attentions": false,
                "output_hidden_states": false,
                "output_past": true,
                "pad_token_id": 0,
                "prefix": null,
                "pruned_heads": {},
                "repetition_penalty": 1.0,
                "task_specific_params": null,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "torchscript": false,
                "type_vocab_size": 2,
                "use_bfloat16": false,
                "vocab_size": 30522
                }
        
        """
        super(BERT, self).__init__()
        self.config = BertConfig(n_labels=n_labels)
        self.bert = BertModel(self.config).from_pretrained(pretrained)
        #output_hidden_states=False,
        #output_attentions=False
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(in_features=768, out_features=n_labels)

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
