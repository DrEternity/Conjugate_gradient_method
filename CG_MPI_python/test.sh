#!/bin/bash

for size in 1000 2000 4000 8000 16000 20000 25000 30000
do
  python3 gen.py $size
  for threads in 1 2 4 6
  do
    echo "SIZE: $size, THREADS: $threads:"
    time mpiexec -n $threads python3 CG.py $size
    echo ""
    echo "///////////////////////////////"
  done
done

