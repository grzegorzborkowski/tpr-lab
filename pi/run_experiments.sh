printf "time,pi,proc,points" >> results_scalable_v2.txt
printf "\n" >> results_scalable_v2.txt
printf "time,pi,proc,points" >> results_nonscalable_v2.txt
printf "\n" >> results_nonscalable_v2.txt

for NUMBER_OF_POINTS in 10000 100000000 500000000
#for NUMBER_OF_POINTS in 10
do
	for NUMBER_OF_PROC in {1..8}
	do
		mpiexec -machinefile allnodes -np $NUMBER_OF_PROC /home/students/g/b/gborkows/pi/mpi_pi_2.py $NUMBER_OF_POINTS scalable >> results_scalable_v2.txt 
		mpiexec -machinefile allnodes -np $NUMBER_OF_PROC /home/students/g/b/gborkows/pi/mpi_pi_2.py $NUMBER_OF_POINTS non-scalable >> results_nonscalable_v2.txt
	done
done

