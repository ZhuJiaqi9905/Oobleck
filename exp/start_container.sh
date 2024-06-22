#!/bin/bash

# run in host
# addrs=(172.31.14.142 172.31.11.170)

addrs=(172.31.10.88 172.31.8.235)

# addrs=(172.31.11.113 172.31.9.213)

image="oobleck:v14"

docker_command="docker start oob-%d"

mkdir -p ./tmp/logs/install/

for addr in "${addrs[@]}"; do
    # ssh ${addr} -o "StrictHostKeyChecking no" "$(printf "${docker_command}" 0 0)
    #     $(printf "${docker_command}" 1)
    #     $(printf "${docker_command}" 2)
    #     $(printf "${docker_command}" 3)" >./tmp/logs/install/sc-${addr}.log 2>&1 &


    ssh ${addr} -o "StrictHostKeyChecking no" "$(printf "${docker_command}" 0 0)
        $(printf "${docker_command}" 1)
        $(printf "${docker_command}" 2)
        $(printf "${docker_command}" 3)
        $(printf "${docker_command}" 4)
        $(printf "${docker_command}" 5)
        $(printf "${docker_command}" 6)
        $(printf "${docker_command}" 7)" >./tmp/logs/install/sc-${addr}.log 2>&1 &     



#    ssh ${addr} -o "StrictHostKeyChecking no" "$(printf "${docker_command}" 0 0)" >./tmp/logs/install/sc-${addr}.log 2>&1 &
done
