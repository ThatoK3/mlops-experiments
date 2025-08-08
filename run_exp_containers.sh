#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN="\033[0;32m"
NC="\033[0m" # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"
}

# Start services
log "Starting MLflow Jupyter..."
bash ./containers/mlflow_jupyter.sh

log "Starting MLflow MySQL..."
bash ./containers/mlflow_mysql.sh

log "Waiting 60 seconds for MySQL to initialize..."
sleep 60

log "Starting MLflow Server..."
bash ./containers/mlflow_server.sh

log "All MLflow services started successfully."
