import torch
import torch.nn as nn
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from transformers import BertModel

from .image_encoder import ImageBertEncoder


class ImageBertEmbeddings(nn.Module):
    def __init__(self, config, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.config = config
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_images):
        token_embeddings = input_images
        seq_length = token_embeddings.size(1)

        # image segment id is 1
        token_type_ids = torch.ones(
            (input_images.size(0), seq_length),
            dtype=torch.long,
            device=input_images.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, config):
        super(MultimodalBertEncoder, self).__init__()
        self.config = config

        bert = BertModel.from_pretrained(config.bert_model_name)
        self.txt_embeddings = bert.embeddings
        self.encoder = bert.encoder
        self.pooler = bert.pooler

        self.img_embeddings = ImageBertEmbeddings(config, self.txt_embeddings)
        self.img_encoder = nn.Linear(1024, config.hidden_size)
        self.head_mask = [None for _ in range(self.config.num_hidden_layers)]

    def forward(self, sample_list):
        input_image = sample_list.img_feature
        input_txt = sample_list.input_ids
        device = input_txt.device

        img = self.img_encoder(input_image)
        img_embed_out = self.img_embeddings(img)
        txt_embed_out = self.txt_embeddings(input_txt, sample_list.segment_ids)
        encoder_input = torch.cat([txt_embed_out, img_embed_out, ], 1)

        input_modal_shape = img_embed_out.size()[:-1]

        attention_mask = torch.cat(
            [
                sample_list.input_mask,
                torch.ones(input_modal_shape, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(encoder_input, extended_attention_mask, head_mask=self.head_mask)

        return self.pooler(encoded_layers[-1])


@registry.register_model("vl_transformer")
class VLTransformer(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/vl_transformer/defaults.yaml"

    def build(self):
        config = self.config
        self.enc = MultimodalBertEncoder(config)
        self.clf = nn.Linear(config.text_hidden_size, config.num_labels)

    def forward(self, sample_list):
        output = {}
        x = self.enc(sample_list)
        logits = self.clf(x)
        reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)
        output["scores"] = reshaped_logits
        return output
