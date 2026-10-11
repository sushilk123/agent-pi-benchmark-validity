#!/usr/bin/env bash
# Fetch the third-party benchmarks into ./benchmarks (not redistributed in this repo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks
[ -d benchmarks/InjecAgent ] || git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git benchmarks/InjecAgent
[ -d benchmarks/BIPIA ]      || git clone --depth 1 https://github.com/microsoft/BIPIA.git benchmarks/BIPIA
echo "Benchmarks ready. AgentDojo is installed via pip (see requirements.txt)."
