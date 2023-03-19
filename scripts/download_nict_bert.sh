#!/usr/bin/env bash

function download() {
    wget https://alaginrc.nict.go.jp/nict-bert/NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip -P downloads
    cd downloads
    unzip NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip
    rm NICT_BERT-base_JapaneseWikipedia_32K_BPE.zip
}

function main() {
    download
}

main
