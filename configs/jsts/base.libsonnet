local task_name = "JSTS";
local metric_name = "stsb";

local max_seq_length = 128;
local num_epochs = 4;
local batch_size = 32;
local learning_rate = 5e-5;
local warmup_ratio = 0.1;
local num_labels = 1;

local num_train_steps = 1560;

local collate_fn = { type: "transformers::DefaultDataCollator" };
local validate_every = null;
local val_metric_name = "pearson";
local devices = 1;

{
    steps(pretrained_model) ::
        {
            raw_data: {
                type: "datasets::load",
                path: "shunk031/JGLUE",
                name: task_name,
            },
            tokenize_data: {
                type: "tokenize_glue",
                metric_name: metric_name,
                dataset: { type: "ref", ref: "raw_data" },
                tokenizer: { pretrained_model_name_or_path: pretrained_model },
                max_seq_length: max_seq_length,
            },
            train_model: {
                type: "torch::train",
                model: {
                    type: "jglue::hf_glue_model",
                    base_model: {
                        type: "transformers::AutoModelForSequenceClassification::from_pretrained",
                        pretrained_model_name_or_path: pretrained_model,
                        num_labels: num_labels,
                    },
                    label_key: "labels",
                    metric_path: "glue",
                    metric_name: metric_name,
                },
                dataset_dict: { type: "ref", ref: "tokenize_data" },
                training_engine: {
                    type: "torch",
                    optimizer: {
                        type: "transformers::AdamW",
                        lr: learning_rate,
                    },
                    lr_scheduler: {
                        type: "transformers::cosine",
                        num_warmup_steps: std.floor(num_train_steps * warmup_ratio),
                        num_training_steps: num_train_steps,
                    },
                },
                train_epochs: num_epochs,
                train_dataloader: {
                    shuffle: true,
                    batch_size: batch_size,
                    collate_fn: collate_fn,
                },
                validation_split: "validation",
                validation_dataloader: {
                    shuffle: false,
                    batch_size: batch_size,
                    collate_fn: collate_fn,
                },
                validate_every: validate_every,
                val_metric_name: val_metric_name,
                minimize_val_metric: false,
                checkpoint_every: num_train_steps,
                device_count: devices,
                callbacks: [
                    {
                        type: "wandb::log",
                        project: "tango-jglue-benchmarks",
                        entity: "shunk031",
                        group: task_name,
                        name: "%s - %s" % [task_name, pretrained_model],
                        tags: [
                            pretrained_model,
                            task_name,
                            metric_name,
                        ],
                    },
                ],
            },
            eval_model: {
                type: "torch::eval",
                model: { type: "ref", ref: "train_model" },
                dataset_dict: { type: "ref", ref: "tokenize_data" },
                dataloader: {
                    shuffle: false,
                    batch_size: batch_size,
                    collate_fn: collate_fn,

                },
                metric_names: [ "loss", "pearson", "spearmanr" ],
                test_split: "validation",
            },
        },
}
