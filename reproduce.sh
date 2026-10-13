#!/usr/bin/env bash
# One-command reproduction of every RetailSense result, in pipeline order.
#   bash reproduce.sh            rebuild everything (LLM answers from cache; misses call the API)
#   bash reproduce.sh --offline  rebuild from the disk cache only; any LLM cache miss is an error
# The final step touches the TEST split and only runs with tau frozen in configs/experiment.yaml.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
OFFLINE=${1:-}
run() { echo; echo "== $*"; MLFLOW_DISABLE_AGENT_HINT=1 $PY -W ignore -m "$@"; }

run retailsense.data.india_generator
run retailsense.data.load
run retailsense.data.weather
run retailsense.data.trends
run retailsense.data.features
run retailsense.eval.run_experiment --phase 2
run retailsense.eval.h1_analysis
run retailsense.eval.deliberation_eval
run retailsense.eval.inventory_eval
run retailsense.eval.run_experiment --final $OFFLINE
run pytest -q
