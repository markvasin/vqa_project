from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from torch import nn

from .attention import StackedAttention

_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("simple_baseline_gqa")
class SimpleBaseline(BaseModel):
    """CNNLSTM + Stack Spatial Attention: A baseline model for VQA task
    Image encoder is a pretrained resnet-101
    Text encoder is a bi-directional LSTM
    Args:
        config (DictConfig): Configuration node containing all of the necessary
                             config required to initialize the model.
    Inputs: sample_list (SampleList)
        - **sample_list** should contain image attribute for image, text for
          question split into word indices, targets for answer scores
    """

    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/simple_baseline/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0
        num_question_choices = registry.get(
            _TEMPLATES["question_vocab_size"].format(self._datasets[0])
        )
        num_answer_choices = registry.get(
            _TEMPLATES["number_of_answers"].format(self._datasets[0])
        )

        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, freeze=True,
                                                  embedding_dim=self.config.text_embedding.embedding_dim)

        self.lstm = nn.LSTM(**self.config.lstm)

        num_stacked_attn = 2
        stacked_attn_dim = 512
        rnn_dim = 2048
        self.stacked_attns = []
        for i in range(num_stacked_attn):
            sa = StackedAttention(rnn_dim, stacked_attn_dim)
            self.stacked_attns.append(sa)
            self.add_module('stacked-attn-%d' % i, sa)

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = num_answer_choices
        self.classifier = ClassifierLayer(
            classifier_config.type, **classifier_config.params
        )

    def forward(self, sample_list):
        self.lstm.flatten_parameters()

        question = sample_list.text
        image_features = sample_list.img_feature

        # Get (h_n, c_n), last hidden and cell state
        _, (hidden, cell) = self.lstm(self.word_embedding(question))
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden = hidden.transpose(0, 1)
        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)

        # Fuse into single dimension
        # fused = torch.cat([hidden, image_features], dim=-1)
        u = hidden
        for sa in self.stacked_attns:
            u = sa(image_features, u)

        scores = self.classifier(u)

        return {"scores": scores}
