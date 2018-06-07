#!/usr/bin/env bash

printf "N,time \n" >> cpu_results.txt
g++ FD_CPU.cpp -O3 -o cpu

for N_size in {992..1600..32};
do
    ./cpu $N_size >> cpu_results.txt
done