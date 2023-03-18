#!/usr/bin/env bash

function download() {
    wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
    tar xf jumanpp-2.0.0-rc3.tar.xz
}

function build() {
    cd jumanpp-2.0.0-rc3/
    mkdir bld
    cd bld
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/.local
}

function install() {
    make install -j 32
}

function cleanup() {
    rm jumanpp-2.0.0-rc3.tar.xz
    rm -rf jumanpp-2.0.0-rc3
}

function main() {
    download
    build
    install
}

main
