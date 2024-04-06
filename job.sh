#!/bin/bash
python -m oobleck.run \
--config_path ./examples/gpt2.yaml \
--node_ips 172.21.0.42 172.21.0.46 172.21.0.90 172.21.0.91 \
--node_port 2222 \
--master_ip 172.21.0.42 \
--master_port 60000