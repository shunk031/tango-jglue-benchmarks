import logging
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Union

import datasets as ds
from tango import Step
from tango.common.params import Params
from tango.format import Format
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer
from tango.step import StepResources

logger = logging.getLogger(__name__)


@Step.register("tokenize_swag")
class TokenizeSwag(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT = DatasetsFormat()

    def __init__(
        self,
        remove_columns: Optional[List[str]] = None,
        step_name: Optional[str] = None,
        cache_results: Optional[bool] = None,
        step_format: Optional[Format] = None,
        step_config: Optional[Union[Dict[str, Any], Params]] = None,
        step_unique_id_override: Optional[str] = None,
        step_resources: Optional[StepResources] = None,
        step_metadata: Optional[Dict[str, Any]] = None,
        step_extra_dependencies: Optional[Iterable["Step"]] = None,
        **kwargs,
    ):
        super().__init__(
            step_name,
            cache_results,
            step_format,
            step_config,
            step_unique_id_override,
            step_resources,
            step_metadata,
            step_extra_dependencies,
            **kwargs,
        )
        self.remove_columns = (
            remove_columns
            if remove_columns
            else [
                "q_id",
                "question",
                "choice0",
                "choice1",
                "choice2",
                "choice3",
                "choice4",
            ]
        )

    def run(  # type: ignore[override]
        self,
        dataset: ds.DatasetDict,
        tokenizer: Tokenizer,
        max_seq_length: int,
        pad_to_max_length: bool,
        choice_prefix: str = "choice",
        context_name: str = "question",
    ) -> ds.DatasetDict:
        ending_names = [f"{choice_prefix}{i}" for i in range(5)]

        if max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(max_seq_length, tokenizer.model_max_length)

        def tokenize_function(examples):
            first_sentences = [[context] * 5 for context in examples[context_name]]
            second_sentences = [
                [f"{examples[end][i]}" for end in ending_names]
                for i in range(len(examples[context_name]))
            ]

            # Flatten out
            first_sentences = list(chain(*first_sentences))
            second_sentences = list(chain(*second_sentences))

            # Tokenize
            tokenized_examples = tokenizer(
                first_sentences,
                second_sentences,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if pad_to_max_length else False,
            )

            # Un-flatten
            return {
                k: [v[i : i + 5] for i in range(0, len(v), 5)]
                for k, v in tokenized_examples.items()
            }

        dataset = dataset.map(
            function=tokenize_function,
            batched=True,
            desc="Tokenizing dataset",
            remove_columns=self.remove_columns,
        )
        return dataset
