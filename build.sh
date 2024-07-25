#!/bin/bash


CC=gcc
CFLAGS="-Wall -Wextra -fanalyzer -g3 -Og"

set -x

${CC} ${CFLAGS} -shared -fPIC src/mlpcore.c -o build/libmlpcore.so
${CC} ${CFLAGS} tests/test01.c -Wl,-rpath=./build -L./build -lmlpcore -o build/test01.o


exit
