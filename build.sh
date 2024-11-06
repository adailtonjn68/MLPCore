#!/bin/bash

project="mlpcore"

CC=gcc
CFLAGS="-Wall -Wextra -fanalyzer -g3 -Og"
LFLAGS="-Wl,-rpath=./build -L./build -l${project} -lm"

TESTFILES=("test01_feedforward" "test02_randomweights" "test03_backpropagation")

set -x

mkdir -p build
${CC} ${CFLAGS} -shared -fPIC src/${project}.c -o build/lib${project}.so

for srcfile in ${TESTFILES[@]}; do
    ${CC} ${CFLAGS} tests/${srcfile}.c ${LFLAGS} -o build/${srcfile}.o
done

exit
