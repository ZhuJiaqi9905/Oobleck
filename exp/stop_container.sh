#!/bin/bash

# run in host
# addrs=(172.31.14.142 172.31.11.170)
addrs=(172.21.0.42 172.21.0.46 172.21.0.90 172.21.0.92)


# Define the common part of the docker run command

docker_command="docker stop oob-%d; docker rm oob-%d;"

mkdir -p ./tmp/logs/install/

for addr in "${addrs[@]}"; do
    ssh ${addr} "$(printf "${docker_command}" 0 0)
        $(printf "${docker_command}" 1 1)
        $(printf "${docker_command}" 2 2)
        $(printf "${docker_command}" 3 3)
        $(printf "${docker_command}" 4 4)
        $(printf "${docker_command}" 5 5)        
        $(printf "${docker_command}" 6 6)
        $(printf "${docker_command}" 7 7)"

done
