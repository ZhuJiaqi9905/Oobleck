#!/bin/bash
for ((i = 0; i < 16; i++)); do
    kubectl delete -f pod${i}.yaml
done