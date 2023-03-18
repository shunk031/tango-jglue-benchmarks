import evaluate
from tango.integrations.torch import Model


@Model.register("jglue::hf_glue_model")
class HfGlueModel(Model):
    def __init__(
        self,
        base_model: Model,
        label_key: str,
        metric_path: str,
        metric_name: str,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.label_key = label_key

        self.metric = evaluate.load(path=metric_path, config_name=metric_name)
        self.is_regression = metric_name == "stsb"

    def forward(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)
        predictions = output.logits
        gold_labels = kwargs.get(self.label_key)

        output_dict = dict(output)

        if gold_labels is not None:
            if self.is_regression:
                # shape: (batch_size, 1) -> (batch_size,)
                predictions = predictions.squeeze(dim=1)
            else:
                # shape: (batch_size, num_labels) -> (batch_size,)
                predictions = predictions.argmax(dim=1)

            result = self.metric.compute(
                predictions=predictions, references=gold_labels
            )
            output_dict.update(result)

        return output_dict
