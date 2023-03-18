local task_name = "JCommonsenseQA";

local max_seq_length = 64;
local num_epochs = 4;
local batch_size = 64;
local learning_rate = 5e-5;
local warmup_ratio = 0.1;
local pad_to_max_length = false;

local num_train_steps = 560;
local validate_every = 100;
local val_metric_name = "accuracy";
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
                type: "tokenize_swag",
                dataset: { type: "ref", ref: "raw_data" },
                tokenizer: { pretrained_model_name_or_path: pretrained_model },
                max_seq_length: max_seq_length,
                pad_to_max_length: pad_to_max_length,
            },
            train_model: {
                type: "torch::train",
                model: {
                    type: "jglue::hf_swag_model",
                    base_model: {
                        type: "transformers::AutoModelForMultipleChoice::from_pretrained",
                        pretrained_model_name_or_path: pretrained_model,
                    },
                    label_key: "labels",
                },
                dataset_dict: { type: "ref", ref: "tokenize_data" },
                training_engine: {
                    type: "torch",
                    optimizer: {
                        type: "torch::AdamW",
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
                    collate_fn: if pad_to_max_length then {
                        type: "transformers::DefaultDataCollator"
                    } else {
                        type: "transformers::DataCollatorForMultipleChoice",
                        tokenizer: { pretrained_model_name_or_path: pretrained_model }
                    },
                },
                validation_split: "validation",
                validation_dataloader: {
                    shuffle: false,
                    batch_size: batch_size,
                    collate_fn: if pad_to_max_length then {
                        type: "transformers::DefaultDataCollator"
                    } else {
                        type: "transformers::DataCollatorForMultipleChoice",
                        tokenizer: { pretrained_model_name_or_path: pretrained_model }
                    },
                },
                validate_every: validate_every,
                val_metric_name: val_metric_name,
                minimize_val_metric: false,
                checkpoint_every: validate_every * 3,
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
                    collate_fn: if pad_to_max_length then {
                        type: "transformers::DefaultDataCollator"
                    } else {
                        type: "transformers::DataCollatorForMultipleChoice",
                        tokenizer: { pretrained_model_name_or_path: pretrained_model }
                    },
                },
                metric_names: [ "loss", "accuracy" ],
                test_split: "validation",
            },
        },
}
