#!/bin/bash
python -m oobleck.run \
--config_path ./examples/bert_340M.yaml \
--node_ips 10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.42 \
--node_ports 2220 2221 2222 2223 \
--master_ip 10.20.23.42 \
--master_port 60000
