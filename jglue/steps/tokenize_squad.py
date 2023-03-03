import logging
from typing import Dict, Optional, Tuple

import datasets as ds
from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer

logger = logging.getLogger(__name__)


@Step.register("tokenize_squad")
class TokenizeSquad(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT = DatasetsFormat()

    def run(  # type: ignore[override]
        self,
        dataset: ds.DatasetDict,
        tokenizer: Tokenizer,
        max_seq_length: int,
    ) -> ds.DatasetDict:
        tng_dataset = dataset["train"]

        def prepare_tng_features(examples):
            breakpoint()

        tng_dataset = tng_dataset.map(
            function=prepare_tng_features,
            desc="Running tokenizer on train dataset",
        )
