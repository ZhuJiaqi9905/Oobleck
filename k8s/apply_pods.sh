#!/bin/bash
for ((i = 0; i < 16; i++)); do
    kubectl apply -f pod${i}.yaml
done