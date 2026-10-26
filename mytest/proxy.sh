python load_balance_proxy_layerwise_server_example.py \
  --host 10.246.63.45 \
  --port 12348 \
  --prefiller-hosts 10.246.63.49 \
  --prefiller-ports 8004 \
  --decoder-hosts 10.246.63.43 10.246.63.43 10.246.63.43 10.246.63.43 10.246.63.43 10.246.63.43 10.246.63.43 10.246.63.43 \
  --decoder-ports 8005 8006 8007 8008 8009 8010 8011 8012