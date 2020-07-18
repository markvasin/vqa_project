from copy import deepcopy

import torch
from torch import nn

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer, ConvNet, Flatten

# Builder methods for image encoder
from mmf.utils.build import build_image_encoder

_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("simple_baseline")
class SimpleBaseline(BaseModel):
    """CNNLSTM is a simple model for vision and language tasks. CNNLSTM is supposed
    to acts as a baseline to test out your stuff without any complex functionality.
    Passes image through a CNN, and text through an LSTM and fuses them using
    concatenation. Then, it finally passes the fused representation from a MLP to
    generate scores for each of the possible answers.
    Args:
        config (DictConfig): Configuration node containing all of the necessary
                             config required to initialize CNNLSTM.
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

        self.text_embedding = nn.Embedding(
            num_question_choices, self.config.text_embedding.embedding_dim
        )
        self.lstm = nn.LSTM(**self.config.lstm)

        self.vision_module = build_image_encoder(self.config.image_encoder)

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
        image = sample_list.image

        # Get (h_n, c_n), last hidden and cell state
        _, hidden = self.lstm(self.text_embedding(question))
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden = hidden[0].transpose(0, 1)

        # X should be 2 so we can merge in that dimension
        assert hidden.size(1) == 2, _CONSTANTS["hidden_state_warning"]

        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)

        image_features = self.vision_module(image)
        # Flatten the embeddings before concatenation
        image_features = torch.flatten(image_features, start_dim=1)

        # Fuse into single dimension
        fused = torch.cat([hidden, image_features], dim=-1)
        scores = self.classifier(fused)

        return {"scores": scores}
