#!/usr/bin/env bash

set -e
set -u


find ./   -type d -name "__pycache__" -exec rm -r {} +

echo "contain clean success"