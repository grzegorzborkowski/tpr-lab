#!/usr/bin/env python
from mpi4py import MPI
import numpy
import random
import sys
from decimal import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main():
	args = sys.argv
	if len(args) < 3:
		print "Invalid number of arguments"
		return
	else:
		number_of_points = int(args[1])
		is_scalable_experiment = str(args[2])
		if not is_scalable_experiment in ['scalable', 'non-scalable']:
			print "Unknown option value. Use scalable or non-scalable only"
			return
		if is_scalable_experiment == 'scalable':
			calculate_pi_with_mpi(number_of_points, True)
		else:
			calculate_pi_with_mpi(number_of_points, False)

def calculate_pi(number_of_points):
	number_of_hits = 0
	
	for point in xrange(number_of_points):
		x_value = random.uniform(0,1)
		y_value = random.uniform(0,1)
		if x_value*x_value + y_value*y_value <= 1:
	 		number_of_hits += 1
	return number_of_hits 

def calculate_pi_with_mpi(number_of_points, is_scalable):
	comm.barrier()
	start = MPI.Wtime()
	if is_scalable:
		pi_result = calculate_pi(number_of_points)
	else:
		pi_result = calculate_pi(number_of_points/comm.Get_size())
	result = comm.gather(pi_result, root=0)
	if rank == 0: # master
		end = MPI.Wtime()
		time = end - start
		number_of_hits = 0
		for partial_result in result:
			number_of_hits += partial_result
		if is_scalable:
			number_of_attempts = number_of_points * comm.Get_size()
		else:
			number_of_attempts = number_of_points
		avg = 4 * Decimal(number_of_hits) / Decimal(number_of_attempts)
		print str(time) + "," + str(avg) + "," + str(comm.Get_size()) + "," + str(number_of_points)
main()

