#!/usr/bin/env bash

printf "N,block,time \n" >> global_results_limits.txt
printf "N,block,time \n" >> shared_results_limits.txt
printf "N,block,time \n" >> texture_results_limits.txt

nvcc FD_2D_global.cu -w -o global

nvcc FD_2D_texture_pad.cu -w -o texture

for N_size in {5596..8200..200};
do
    for BLOCK_size in 16
    do
#        nvcc FD_2D_shared.cu -w -o shared -DBSZ=$BLOCK_size
        ./global $N_size $BLOCK_size >> global_results_limits.txt
#         ./shared $N_size $BLOCK_size >> shared_results_limits.txt
#        ./texture $N_size $BLOCK_size >> texture_results_limits.txt
    done

done