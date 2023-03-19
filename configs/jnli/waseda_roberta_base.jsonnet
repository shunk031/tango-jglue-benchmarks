local jnli = import "base.libsonnet";

{
    steps: jnli.steps(
        "nlp-waseda/roberta-base-japanese",
        is_pre_tokenize=true,
        analyzer="jumanpp",
    ),
}
