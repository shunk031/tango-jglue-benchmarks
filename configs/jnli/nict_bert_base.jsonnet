local jnli = import "base.libsonnet";

{
    steps: jnli.steps(
        "./downloads/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
        is_pre_tokenize=true,
        analyzer="mecab",
    )
}
