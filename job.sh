#!/bin/bash
python -m oobleck.run \
--config_path ./examples/gpt3_6_7B.yaml \
--node_ips 10.20.23.42 10.20.23.46 10.20.23.90 10.20.23.92 10.20.23.91 \
--node_port 2222 \
--master_ip 10.20.23.42 \
--master_port 60000