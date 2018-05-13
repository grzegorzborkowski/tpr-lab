printf "time,number_of_points,number_of_buckets,range_of_numbers,number_of_threads" >> results_scalable.txt
printf "\n" >> results_scalable.txt

#####
printf "time,number_of_points,number_of_buckets,range_of_numbers,number_of_threads" >> results_merging_scalable.txt
printf "\n" >> results_merging_scalable.txt

#####
printf "time,number_of_points,number_of_buckets,range_of_numbers,number_of_threads" >> results_non-scalable.txt
printf "\n" >> results_non-scalable.txt

#####
printf "time,number_of_points,number_of_buckets,range_of_numbers,number_of_threads" >> results_merging_non-scalable.txt
printf "\n" >> results_merging_non-scalable.txt

make

for NUMBER_OF_POINTS in 1000000 10000000
do
        for NUMBER_OF_BUCKETS in 100 1000 10000
        do
            for RANGE_OF_NUMBERS in 100 1000 10000
            do
                 for NUMBER_OF_THREADS in {1..8}
                    do
                        ./sorting $NUMBER_OF_POINTS $NUMBER_OF_BUCKETS $RANGE_OF_NUMBERS $NUMBER_OF_THREADS scalable >> results_scalable.txt
                        ./sorting $NUMBER_OF_POINTS $NUMBER_OF_BUCKETS $RANGE_OF_NUMBERS $NUMBER_OF_THREADS non-scalable >> results_non-scalable.txt
                        ./sorting_merging $NUMBER_OF_POINTS $NUMBER_OF_BUCKETS $RANGE_OF_NUMBERS $NUMBER_OF_THREADS scalable >> results_merging_scalable.txt
                        ./sorting_merging $NUMBER_OF_POINTS $NUMBER_OF_BUCKETS $RANGE_OF_NUMBERS $NUMBER_OF_THREADS non-scalable >> results_merging_non-scalable.txt
                    done
            done
        done
done
