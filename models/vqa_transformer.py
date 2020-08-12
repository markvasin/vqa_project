import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import BertConfig
from transformers.modeling_bert import BertEncoder, BertPooler, BertPredictionHeadTransform

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from .image_encoder import ImageBertEncoder


@registry.register_model("vqa_transformer")
class VqaTransformer(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/vqa_transformer/defaults.yaml"

    def build(self):
        self.text_processor = registry.get(self._datasets[0] + "_text_processor")
        self.vocab = self.text_processor.vocab
        self.word_embedding = self.vocab.get_embedding(torch.nn.Embedding, freeze=False,
                                                       embedding_dim=self.config.text_embedding.embedding_dim)
        self.segment_embeddings = nn.Embedding(self.config.num_segment_type, self.config.hidden_size)

        self.cls_project = nn.Linear(self.config.text_embedding.embedding_dim, self.config.hidden_size)
        self.lstm = nn.LSTM(**self.config.lstm)
        self.lstm_proj = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        # self.img_encoder = ImageBertEncoder(self.config)
        self.img_proj = nn.Linear(self.config.image_hidden_size, self.config.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        self.transformer = BertEncoder(self.bert_config)
        self.pooler = BertPooler(self.bert_config)

        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.config),
            nn.Linear(self.config.hidden_size, self.config.num_labels),
        )

        self.head_mask = [None for _ in range(self.config.num_hidden_layers)]

    def forward(self, sample_list):
        output = {}
        batch_size = sample_list.text.shape[0]
        device = sample_list.text.device

        question = sample_list.text
        image_features = self.img_proj(sample_list.image_feature_0)

        cls_token_id = torch.tensor(self.vocab.vocab.stoi['[CLS]'], device=device).repeat(batch_size, 1)
        cls_token_embeds = self.word_embedding(cls_token_id)
        cls_token = self.cls_project(cls_token_embeds)
        cls_type_ids = torch.zeros(cls_token.size()[:-1], dtype=torch.long, device=device)
        cls_type_embedding = self.segment_embeddings(cls_type_ids)
        cls_embeddings = cls_token + cls_type_embedding

        text_feat = self.word_embedding(question)
        text_tokens, _ = self.lstm(text_feat)
        text_tokens = self.lstm_proj(text_tokens)
        text_type_ids = torch.zeros(text_tokens.size()[:-1], dtype=torch.long, device=device)
        text_type_embedding = self.segment_embeddings(text_type_ids)
        text_embeddings = text_tokens + text_type_embedding

        img_tokens = image_features # self.img_encoder(image_features)
        img_type_ids = torch.ones(img_tokens.size()[:-1], dtype=torch.long, device=device)
        img_type_embedding = self.segment_embeddings(img_type_ids)
        img_embeddings = img_tokens + img_type_embedding

        embeddings = torch.cat([cls_embeddings, text_embeddings, img_embeddings], 1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # attention mask
        cls_mask = torch.ones(cls_token.size()[:-1], device=device, dtype=torch.long)
        text_mask = sample_list.text_mask
        img_mask = torch.ones(img_tokens.size()[:-1], device=device, dtype=torch.long)
        attention_mask = torch.cat([cls_mask, text_mask, img_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        transformer_outputs = self.transformer(embeddings, extended_attention_mask, head_mask=self.head_mask)
        sequence_output = transformer_outputs[0]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)
        output["scores"] = reshaped_logits

        return output

