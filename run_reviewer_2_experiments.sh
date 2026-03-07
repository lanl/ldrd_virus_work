#!/bin/bash

for i in {0..9}; do
    mkdir "run_$i"
    cd "run_$i"
    echo "Starting trial $i WITH kmers -----"
    time python ../viral_seq/run_workflow.py -tr Mollentze_Training_Fixed_shuffled_$i.csv -ts Mollentze_Holdout_Fixed_shuffled_$i.csv -tc "Human Host" -c "extract" -o none -n 16 -cp 1
    echo "Finished trial $i WITH kmers -----"
    echo "Starting trial $i NO kmers -----"
    time python ../viral_seq/run_workflow.py -tr Mollentze_Training_Fixed_shuffled_$i.csv -ts Mollentze_Holdout_Fixed_shuffled_$i.csv -tc "Human Host" -c "extract" -o none -fs skip -f 0 -n 16 -cp 1
    echo "Finished trial $i NO kmers -----"
    cd ..
done
