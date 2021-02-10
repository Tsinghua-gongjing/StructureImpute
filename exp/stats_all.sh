#!/bin/bash

for p in `cat data/processed/all.list`
do 
    r=$(tail -1 exp/$p/out/log.txt)
    echo $p,$r
done
