#!/bin/bash

for ((i = 2; i < 16; i++)); do
    cp pod0.yaml pod${i}.yaml
done
