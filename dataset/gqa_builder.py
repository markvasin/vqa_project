from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder

from .gqa_dataset import GQADatasetV2


@registry.register_builder("gqa2")
class GQABuilderV2(MMFDatasetBuilder):
    def __init__(self, dataset_name="gqa2", dataset_class=GQADatasetV2, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = GQADatasetV2

    @classmethod
    def config_path(cls):
        return "configs/datasets/gqa2/defaults.yaml"

    def build(self, config, dataset_type="train", *args, **kwargs):
        pass

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor"):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        if hasattr(self.dataset, "answer_processor"):
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )
