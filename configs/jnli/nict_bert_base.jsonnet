local jnli = import "base.libsonnet";

{
    steps: jnli.steps(
        "./downloads/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
        is_pre_tokenize=true,
        analyzer="mecab",
        mecab_dic_dir="/var/lib/mecab/dic/juman-utf8/",
    )
}
