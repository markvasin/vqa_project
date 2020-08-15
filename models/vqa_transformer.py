from importlib import import_module

import torch
import torch.nn as nn
from mmf.datasets.builders.clevr.dataset import CLEVRDataset
from omegaconf import OmegaConf
from transformers import BertConfig
from transformers.modeling_bert import BertEncoder, BertPooler, BertPredictionHeadTransform

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from .image_encoder import ImageBertEncoder, ImageClevrEncoder, ResNet101ImageEncoder
from .mca import MCA_ED, AttFlat, LayerNorm


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
        # clever_word_emb = registry.mapping['state']['clevr_word_embedding']
        # self.vocab = self.text_processor.vocab
        # self.word_embedding = self.vocab.get_embedding(torch.nn.Embedding, freeze=False,
        #                                                embedding_dim=self.config.text_embedding.embedding_dim)

        # self.vocab = registry.mapping['state']['clevr_token_to_index']
        # registry.unregister('clevr_word_embedding')
        # registry.unregister('clevr_token_to_index')
        self.word_embedding = nn.Embedding(
            num_embeddings=83,
            embedding_dim=300
        )

        self.word_embedding.weight.data.copy_(torch.from_numpy(CLEVRDataset.pretrained_emb))

        # self.segment_embeddings = nn.Embedding(self.config.num_segment_type, self.config.hidden_size)

        # self.cls_project = nn.Linear(self.config.text_embedding.embedding_dim, self.config.hidden_size)
        self.lstm = nn.LSTM(**self.config.lstm)
        # self.lstm_proj = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        # self.img_encoder = ResNet101ImageEncoder(self.config)
        self.img_proj = nn.Linear(self.config.image_hidden_size, self.config.hidden_size)

        # self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        # self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        #
        # self.bert_config = BertConfig.from_dict(
        #     OmegaConf.to_container(self.config, resolve=True)
        # )
        # self.transformer = BertEncoder(self.bert_config)
        # self.pooler = BertPooler(self.bert_config)

        # self.classifier = nn.Sequential(
        #     BertPredictionHeadTransform(self.config),
        #     nn.Linear(self.config.hidden_size, 28),
        # )

        # self.head_mask = [None for _ in range(self.config.num_hidden_layers)]

        self.backbone = MCA_ED(CfgLoader)
        # Flatten to vector
        self.attflat_img = AttFlat(CfgLoader)
        self.attflat_lang = AttFlat(CfgLoader)

        self.proj_norm = LayerNorm(CfgLoader.FLAT_OUT_SIZE)
        self.proj = nn.Linear(CfgLoader.FLAT_OUT_SIZE, 28)

    def forward(self, sample_list):
        output = {}
        batch_size = sample_list.text.shape[0]
        device = sample_list.text.device

        question = sample_list.text
        # ques_mask = sample_list.text_mask
        # lang_feat_mask = make_mask(question.unsqueeze(2))

        image = sample_list.image

        # cls_token_id = torch.tensor(CLEVRDataset.token_to_ix['CLS'], device=device).repeat(batch_size, 1)
        # cls_token_embeds = self.word_embedding(cls_token_id)
        # cls_embeddings = self.cls_project(cls_token_embeds)

        text_feat = self.word_embedding(question)
        text_tokens, _ = self.lstm(text_feat)
        # text_tokens = self.lstm_proj(text_tokens)
        # text_type_ids = torch.zeros(text_tokens.size()[:-1], dtype=torch.long, device=device)
        # text_type_embedding = self.segment_embeddings(text_type_ids)
        # text_embeddings = text_tokens #+ text_type_embedding

        img_tokens = self.img_proj(image)
        # img_type_ids = torch.ones(img_tokens.size()[:-1], dtype=torch.long, device=device)
        # img_type_embedding = self.segment_embeddings(img_type_ids)
        # img_embeddings = img_tokens# + img_type_embedding

        # embeddings = torch.cat([cls_embeddings, text_embeddings, img_embeddings], 1)
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)

        # attention mask
        # cls_mask = torch.ones(cls_embeddings.size()[:-1], device=device, dtype=torch.long)
        # text_mask = sample_list.text_mask
        # img_mask = torch.ones(img_tokens.size()[:-1], device=device, dtype=torch.long)
        # attention_mask = torch.cat([cls_mask, text_mask, img_mask], dim=1)
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # transformer_outputs = self.transformer(embeddings, extended_attention_mask, head_mask=self.head_mask)
        # sequence_output = transformer_outputs[0]
        # pooled_output = self.pooler(sequence_output)
        # pooled_output = self.dropout(pooled_output)

        lang_feat_mask = make_mask(question.unsqueeze(2))
        img_feat_mask = make_mask(img_tokens)
        lang_feat, img_feat = self.backbone(
            text_tokens,
            img_tokens,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        # logits = self.classifier(pooled_output)


        # reshaped_logits = logits.contiguous().view(-1, 28)
        output["scores"] = proj_feat

        return output


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class CfgLoader:

    LAYER = 6
    HIDDEN_SIZE = 512
    FF_SIZE = 2048
    MULTI_HEAD = 8
    DROPOUT_R = 0.1
    FLAT_MLP_SIZE = 512
    FLAT_GLIMPSES = 1
    FLAT_OUT_SIZE = 1024
