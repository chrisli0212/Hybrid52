#!/bin/bash
mkdir -p /workspace/.persist/windsurf-server
mkdir -p /workspace/.persist/codeium

if [ -d /root/.windsurf-server ] && [ ! -L /root/.windsurf-server ]; then
    cp -a /root/.windsurf-server/* /workspace/.persist/windsurf-server/ 2>/dev/null || true
fi
if [ -d /root/.codeium ] && [ ! -L /root/.codeium ]; then
    cp -a /root/.codeium/* /workspace/.persist/codeium/ 2>/dev/null || true
fi

rm -rf /root/.windsurf-server
ln -s /workspace/.persist/windsurf-server /root/.windsurf-server

rm -rf /root/.codeium
ln -s /workspace/.persist/codeium /root/.codeium

echo "✓ Windsurf persistence restored"
