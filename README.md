ParallelFuzzyRoughSetApproximationonMPI
=======================================

Parallel Fuzzy Rough Set Approximation on MPI
This code can only run on a Linux environment. It is required that the same path where the code is executing exist on all the nodes. For instance, if the code is getting executed from /home/user/fuzzy/ then this folder must exist on all nodes. 

To compile the code on an Intel cluster, run the below:
mpiicc final_code.c -o fuzz

To compile the code on an Intel cluster, run the below:
mpicc final_code.c -o fuzz

The above will create an executable file "fuzz" which is the MPI program. To execute this program, do the following:

./FRA.sh data_set_dir num_of_ranks num_of_rows num_of_threads

example: ./FRA.sh /home/user/testcase.txt 9 100 4
This example runs the program to process the 100 rows "testcase.txt" file with 9 nodes (1 as master and 8 as slaves) and 4 parallel threads per node.

Note that FRA.sh assumes that you have a hosts file stored in the same location as the FRA.sh and named as "mpd.hosts".
