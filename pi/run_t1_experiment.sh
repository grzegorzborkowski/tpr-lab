printf "time,pi,proc,points" >> results_scalable_t1.txt
printf "\n" >> results_scalable_t1.txt
printf "time,pi,proc,points" >> results_nonscalable_t1.txt
printf "\n" >> results_nonscalable_t1.txt

for NUMBER_OF_POINTS in 10000 100000000 500000000
do
	for NUMBER_OF_TRIES in {1..10}
	do
		mpiexec -machinefile allnodes -np 1 /home/students/g/b/gborkows/pi/mpi_pi_2.py $NUMBER_OF_POINTS scalable >> results_scalable_t1.txt 
		mpiexec -machinefile allnodes -np 1 /home/students/g/b/gborkows/pi/mpi_pi_2.py $NUMBER_OF_POINTS non-scalable >> results_nonscalable_t1.txt
	done
done

