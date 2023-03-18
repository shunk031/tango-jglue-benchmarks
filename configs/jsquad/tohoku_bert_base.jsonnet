local pretrained_model = "cl-tohoku/bert-base-japanese-v2";

local task_name = "JSQuAD";
local metric_name = "squad";

local max_seq_length = 384;
local num_epochs = 4;
local batch_size = 32;
local learning_rate = 5e-5;
local warmup_ratio = 0.1;
local num_labels = 3;

local num_train_steps = 2512;

local collate_fn = { type: "transformers::DefaultDataCollator" };

local validate_every = 100;
local val_metric_name = "accuracy";
local devices = 1;

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "shunk031/JGLUE",
            name: task_name,
        },
        tokenize_data: {
            type: "tokenize_squad",
            dataset: { type: "ref", ref: "raw_data" },
            tokenizer: { pretrained_model_name_or_path: pretrained_model },
            max_seq_length: max_seq_length,
            doc_stride: 128,
            pad_to_max_length: true,
        },
    },
}
