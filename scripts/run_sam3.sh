#!/bin/bash

SESSION_NAME="sam3"
REPO_ROOT="${OPENFRONTIER_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV=sam3

# Start a new tmux session in detached mode
tmux new-session -d -s $SESSION_NAME

# Send commands to the tmux session
tmux send-keys -t $SESSION_NAME "source ../miniconda3/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" C-m
tmux send-keys -t $SESSION_NAME "cd $REPO_ROOT" C-m
tmux send-keys -t $SESSION_NAME "python sam3_server.py" C-m

echo "Started SAM3 server in tmux session '$SESSION_NAME'. To attach, use: tmux attach -t $SESSION_NAME"
