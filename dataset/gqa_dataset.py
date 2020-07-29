import os

import h5py
import json
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.configuration import get_mmf_env

img = None
img_info = {}


def gqa_feature_loader():
    global img, img_info
    if img is not None:
        return img, img_info
    path = os.path.join(get_mmf_env("data_dir"), "datasets", "gqa", "defaults", "features")
    h = h5py.File(f'{path}/gqa_spatial.hdf5', 'r')
    img = h['features']
    img_info = json.load(open(f'{path}/gqa_spatial_merged_info.json', 'r'))
    return img, img_info


class GQADatasetV2(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("gqa2", config, dataset_type, imdb_file_index, *args, **kwargs)
        self.img, self.img_info = gqa_feature_loader()

    def build_features_db(self):
        pass

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        text_processor_argument = {"text": sample_info["question_str"]}
        processed_question = self.text_processor(text_processor_argument)
        current_sample.text = processed_question["text"]
        if "input_ids" in processed_question:
            current_sample.update(processed_question)

        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            idx = int(self.img_info[str(sample_info["image_id"])]['index'])
            current_sample.img_feature = torch.from_numpy(self.img[idx])


        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)
        return current_sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)

            predictions.append(
                {"questionId": question_id.item(), "prediction": answer, }
            )

        return predictions
