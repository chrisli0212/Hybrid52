#!/bin/bash
# Robust version - works even if no requirements.txt or venv is fresh
VENV_PIP="/workspace/venv/bin/pip"

if [ ! -f "$VENV_PIP" ]; then
    echo "⚠️  venv pip missing — bootstrapping..."
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
fi

if [ $# -eq 0 ]; then
    echo "📦 Installing from requirements.txt (or skipping if missing)..."
    if [ -f /workspace/requirements.txt ]; then
        $VENV_PIP install -r /workspace/requirements.txt -q
    else
        echo "→ No requirements.txt yet — nothing to install"
    fi
else
    echo "📦 Installing packages: $@"
    $VENV_PIP install "$@" -q
    $VENV_PIP freeze > /workspace/requirements.txt
    echo "✓ requirements.txt updated"
fi
