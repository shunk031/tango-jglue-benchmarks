import logging
from typing import Dict, Optional, Tuple

import datasets as ds
from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer

logger = logging.getLogger(__name__)

TASK_TO_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@Step.register("tokenize_glue")
class TokenizeGlue(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT = DatasetsFormat()

    def run(  # type: ignore[override]
        self,
        dataset: ds.DatasetDict,
        tokenizer: Tokenizer,
        metric_name: str,
        max_seq_length: int,
    ) -> ds.DatasetDict:
        sentence1_key, sentence2_key = TASK_TO_KEYS[metric_name]

        if max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(max_seq_length, tokenizer.model_max_length)

        def tokenize_function(examples):
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(
                *args,
                truncation=True,
                padding="max_length",
                max_length=max_seq_length,
            )
            result["label"] = examples["label"]

            return result

        dataset = dataset.map(
            function=tokenize_function,
            batched=True,
            desc="Tokenizing dataset",
        )

        return dataset
