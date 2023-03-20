local jnli = import "base.libsonnet";

{
    steps: jnli.steps(
        "nlp-waseda/roberta-large-japanese-with-auto-jumanpp", 
        max_seq_length=128
    )
}
