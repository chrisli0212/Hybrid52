#!/bin/bash
echo "→ Restoring Cursor full AI memory (chats + state.vscdb)..."

# Force-create the exact folders where real memory lives
mkdir -p /workspace/.persist/cursor-config/User/workspaceStorage
mkdir -p /workspace/.persist/cursor-config/User/globalStorage
mkdir -p /workspace/.persist/cursor-server

# Copy chat history specifically (this was missing before)
if [ -d "/root/.config/Cursor/User/workspaceStorage" ]; then
    mkdir -p /workspace/.persist/cursor-config/User/workspaceStorage
    cp -a /root/.config/Cursor/User/workspaceStorage/* /workspace/.persist/cursor-config/User/workspaceStorage/ 2>/dev/null || true
fi

# Full top-level copy (for any other Cursor data)
if [ -d /root/.config/Cursor ] && [ ! -L /root/.config/Cursor ]; then
    cp -a /root/.config/Cursor/* /workspace/.persist/cursor-config/ 2>/dev/null || true
fi

if [ -d /root/.cursor-server ] && [ ! -L /root/.cursor-server ]; then
    cp -a /root/.cursor-server/* /workspace/.persist/cursor-server/ 2>/dev/null || true
fi

# Re-apply symlinks (this is what survives pod changes)
rm -rf /root/.config/Cursor 2>/dev/null || true
ln -sfn /workspace/.persist/cursor-config /root/.config/Cursor

rm -rf /root/.cursor-server 2>/dev/null || true
ln -sfn /workspace/.persist/cursor-server /root/.cursor-server

echo "✓ Cursor full persistence restored (including workspaceStorage & state.vscdb)"
