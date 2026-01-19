#!/bin/bash
# Load API key from .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

export PYTHONPATH="$SCRIPT_DIR:$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"
exec /usr/bin/python3 mcp_server.py
