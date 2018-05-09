printf "time,number_of_points,number_of_buckets,range_of_numbers,number_of_threads" >> results.txt
printf "\n" >> results.txt

make 

for NUMBER_OF_POINTS in 1000000 5000000 10000000 50000000
do
        for NUMBER_OF_BUCKETS in 100 1000 10000
        do
            for RANGE_OF_NUMBERS in 1000 10000 100000
            do
                 for NUMBER_OF_THREADS in {1..8}
                    do
                        ./sorting $NUMBER_OF_POINTS $NUMBER_OF_BUCKETS $RANGE_OF_NUMBERS $NUMBER_OF_THREADS >> results.txt
                    done
            done
        done
done
