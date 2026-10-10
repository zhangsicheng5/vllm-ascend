#!/bin/bash

set -e

curl -f -X POST http://10.246.63.47:7350/start_profile
echo "Profiling started"

sleep 2

curl -f -X POST http://10.246.63.47:7350/stop_profile
echo "Profiling stopped"
