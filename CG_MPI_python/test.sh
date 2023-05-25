#!/bin/bash

for size in 8000 10000 
do
  python gen.py $size
  for threads in 1 2 4 6
  do
    echo "SIZE: $size, THREADS: $threads:"
    time mpiexec -n $threads python3 CG.py $size
    echo ""
    echo "///////////////////////////////"
  done
done

