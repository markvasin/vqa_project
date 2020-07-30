from copy import deepcopy

import torch
import torch.nn.functional as F
from block.models.networks.fusions.fusions import Tucker
from torch import nn
from torch.nn.init import xavier_uniform_

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer

_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)
        self.tucker = Tucker((2048, 2048), 1, mm_dim=50, shared=True)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        s_matrix = (mem * know)
        s_matrix = s_matrix.view(-1, 2048)
        attn = self.tucker([s_matrix, control[-1].repeat(know.size(2), 1)]).view(know.size(2), know.size(0))
        attn = attn.transpose(0, 1)
        attn = F.softmax(attn, 1).unsqueeze(1)
        read = (attn * know).sum(2)
        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12, self_attention=False, memory_gate=False, dropout=0.15):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory


@registry.register_model("mac_network")
class MACNetwork(BaseModel):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/mac_network/defaults.yaml"

    def build(self):
        num_answer_choices = registry.get(_TEMPLATES["number_of_answers"].format(self._datasets[0]))

        self.dim = self.config.dimension

        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, freeze=True,
                                                  embedding_dim=self.config.text_embedding.embedding_dim)

        self.lstm = nn.LSTM(**self.config.lstm)
        self.lstm_proj = nn.Linear(self.dim * 2, self.dim)

        self.mac = MACUnit(**self.config.mac_unit)

        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.out_dim = num_answer_choices
        self.classifier = ClassifierLayer(classifier_config.type, **classifier_config.params)

    def forward(self, sample_list):
        question = sample_list.text
        image_features = sample_list.img_feature

        b_size = question.size(0)
        img = image_features.view(b_size, self.dim, -1)

        embed = self.word_embedding(question)
        lstm_out, (hidden, _) = self.lstm(embed)
        lstm_out = self.lstm_proj(lstm_out)
        hidden = hidden.permute(1, 0, 2).contiguous().view(b_size, -1)

        memory = self.mac(lstm_out, hidden, img)

        out = torch.cat([memory, hidden], 1)
        scores = self.classifier(out)

        return {"scores": scores}
