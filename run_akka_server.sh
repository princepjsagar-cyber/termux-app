#!/bin/bash

# Supervisor script for Akka-like bot server
# Keeps the server running and restarts on crashes

echo "Starting Akka-like bot server supervisor..."

while true; do
    echo "$(date): Starting Akka-like bot server..."
    python3 /workspace/akka_bot_server.py
    
    echo "$(date): Server stopped, restarting in 5 seconds..."
    sleep 5
done