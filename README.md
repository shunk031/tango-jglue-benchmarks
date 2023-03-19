# [`JGLUE`](https://github.com/yahoojapan/JGLUE) benchmarks with [`AI2-Tango`](https://github.com/allenai/tango)

## Setup

- setup the python environment

```console
pip install -U pip wheel setuptools poetry
poetry install
```

- install jumanpp (2.0.0-rc3)

```console
bash scripts/install_jumanpp.sh
```

- download NICT BERT base

```console
bash scripts/download_nict_bert.sh
```

## Train

### MARC-ja

- Tohoku BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/marc_ja/tohoku_bert_base.jsonnet -i jglue/ -w workspace/marc_ja/tohoku_bert_base
```

- Tohoku BERT base (char)

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/marc_ja/tohoku_bert_base_char.jsonnet -i jglue/ -w workspace/marc_ja/tohoku_bert_base_char
```

- LUKE base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/marc_ja/luke_japanese_base_lite.jsonnet -i jglue/ -w workspace/marc_ja/luke_japanese_base_lite
```

- NICT BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/marc_ja/nict_bert_base.jsonnet -i jglue/ -w workspace/marc_ja/nict_bert_base
```

### JSTS

- Tohoku BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jsts/tohoku_bert_base.jsonnet -i jglue/ -w workspace/jsts/tohoku_bert_base
```

- Tohoku BERT base (char)

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jsts/tohoku_bert_base_char.jsonnet -i jglue/ -w workspace/jsts/tohoku_bert_base_char
```

- LUKE base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jsts/luke_japanese_base_lite.jsonnet -i jglue/ -w workspace/jsts/luke_japanese_base_lite
```

- NICT BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jsts/nict_bert_base.jsonnet -i jglue/ -w workspace/jsts/nict_bert_base
```

### JNLI

- Tohoku BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jnli/tohoku_bert_base.jsonnet -i jglue/ -w workspace/jnli/tohoku_bert_base
```

- Tohoku BERT base (char)

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jnli/tohoku_bert_base_char.jsonnet -i jglue/ -w workspace/jnli/tohoku_bert_base_char
```

- LUKE base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jnli/luke_japanese_base_lite.jsonnet -i jglue/ -w workspace/jnli/luke_japanese_base_lite
```

- NICT BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jnli/nict_bert_base.jsonnet -i jglue/ -w workspace/jnli/nict_bert_base
```

### JSQuAD

WIP

### JCommonsenseQA

- Tohoku BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jcommonsenseqa/tohoku_bert_base.jsonnet -i jglue/ -w workspace/jcommonsenseqa/tohoku_bert_base
```

- Tohoku BERT base (char)

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jcommonsenseqa/tohoku_bert_base_char.jsonnet -i jglue/ -w workspace/jcommonsenseqa/tohoku_bert_base_char
```

- LUKE base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jcommonsenseqa/luke_japanese_base_lite.jsonnet -i jglue/ -w workspace/jcommonsenseqa/luke_japanese_base_lite
```

- NICT BERT base

```console
CUDA_VISIBLE_DEVICES=0 tango run configs/jcommonsenseqa/nict_bert_base.jsonnet -i jglue/ -w workspace/jcommonsenseqa/nict_bert_base
```
