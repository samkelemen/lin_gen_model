#!/bin/bash

ids=(1 2 3 5 6 7 8 10 11 13 14 15 16 17 19 20 22 23 24 25 26 27 28 29 31)
for id in "${ids[@]}"; do
    sbatch model.sh $id
done