from typing import Any, Dict, Optional

import evaluate
from tango.common.dataset_dict import DatasetDictBase
from tango.integrations.torch import TrainCallback
from tango.integrations.torch.data import DataLoader
from tango.integrations.torch.train_config import TrainConfig
from tango.integrations.torch.training_engine import TrainingEngine
from tango.workspace import Workspace


@TrainCallback.register("jglue::calculate_val_metric_callback")
class CalculateValMetricCallback(TrainCallback):
    def __init__(
        self,
        metric_path: str,
        metric_name: str,
        workspace: Workspace,
        train_config: TrainConfig,
        training_engine: TrainingEngine,
        dataset_dict: DatasetDictBase,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> None:
        super().__init__(
            workspace,
            train_config,
            training_engine,
            dataset_dict,
            train_dataloader,
            validation_dataloader,
        )
        self.metric = evaluate.load(path=metric_path, config_name=metric_name)

    def pre_val_batch(
        self, step: int, val_step: int, epoch: int, val_batch: Dict[str, Any]
    ) -> None:
        breakpoint()

    def post_val_batch(
        self, step: int, val_step: int, epoch: int, val_batch_outputs: Dict[str, Any]
    ) -> None:
        breakpoint()
