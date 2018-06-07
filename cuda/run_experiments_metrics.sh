#!/usr/bin/env bash

nvcc FD_2D_global.cu -w -o global
nvcc FD_2D_texture_pad.cu -w -o texture

#printf "N,block,update_min_occupancy,update_max_occupancy,update_avg_occupancy,copy_min_occupancy,copy_max_occupancy,copy_avg_occupancy \n" >> metryki_global.txt
#printf "N,block,update_min_occupancy,update_max_occupancy,update_avg_occupancy,copy_min_occupancy,copy_max_occupancy,copy_avg_occupancy \n" >> metryki_shared.txt
#printf "N,block,update_min_occupancy,update_max_occupancy,update_avg_occupancy,copy_min_occupancy,copy_max_occupancy,copy_avg_occupancy \n" >> metryki_texture.txt

prof_and_transform() {
    printf "$2,$3," >> $4
    nvprof --metrics achieved_occupancy $1 $2 $3 2> tmp.txt &&
    tail -n 4 tmp.txt > tmp2.txt && sed -n '2p;4p' < tmp2.txt | awk '{print $5,$6,$7}' | tr '\n' ' ' | tr ' ' ',' | head -c -1 >> $4
    printf "\n" >> $4
}

for N in {2080..4096..32};
do
    for BLOCK in 8 16 24 32
    do
            nvcc FD_2D_shared.cu -w -o shared -DBSZ=$BLOCK
            prof_and_transform ./global $N $BLOCK metryki_global.txt
            prof_and_transform ./shared $N $BLOCK metryki_shared.txt
            prof_and_transform ./texture $N $BLOCK metryki_texture.txt
    done
done
