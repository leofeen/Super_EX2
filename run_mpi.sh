#! /usr/bin/bash

for i in {1..4}; do
    n=$((2**i))
    bsub < ./jobs/job_mpi_400_600_"$n".lsf
done

for i in {2..5}; do
    n=$((2**i))
    bsub < ./jobs/job_mpi_800_1200_"$n".lsf
done