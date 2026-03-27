#!/bin/bash

# Windsurf Chat History Persistence (Symlink Method)
# This script ensures Windsurf AI chat memory persists across RunPod instance changes

echo "→ Setting up Windsurf chat memory persistence..."

# Ensure persistent directories exist
mkdir -p /workspace/.codeium_persist
mkdir -p /workspace/.persist/codeium

# Remove any existing .codeium directory that's not a symlink
if [ -d /root/.codeium ] && [ ! -L /root/.codeium ]; then
    echo "→ Backing up existing .codeium directory..."
    cp -a /root/.codeium/* /workspace/.codeium_persist/ 2>/dev/null || true
    rm -rf /root/.codeium
fi

# Create symlink from /root/.codeium to persistent storage
ln -sfn /workspace/.codeium_persist /root/.codeium

# Also ensure the .persist/codeium path is symlinked for consistency
ln -sfn /workspace/.codeium_persist /workspace/.persist/codeium

echo "✅ Windsurf chat memory will now persist at /workspace/.codeium_persist"
echo "   Symlinked: /root/.codeium → /workspace/.codeium_persist"
