#!/bin/bash

set -e

curl -f -X POST http://localhost:8121/start_profile
echo "Profiling started"

sleep 2

curl -f -X POST http://localhost:8121/stop_profile
echo "Profiling stopped"
