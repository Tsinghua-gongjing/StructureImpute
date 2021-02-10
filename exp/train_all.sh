#!/bin/bash

for p in `cat data/processed/all.list`
do 
    exp/$p/train.sh &
done
