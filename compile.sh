#!/bin/sh
fi=$1
fo=$2

g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result $fi -o $fo -lgsl -lm -lgslcblas
