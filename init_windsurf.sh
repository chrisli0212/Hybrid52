#!/bin/bash
mkdir -p /workspace/.persist/windsurf-server
mkdir -p /workspace/.codeium_persist

if [ -d /root/.windsurf-server ] && [ ! -L /root/.windsurf-server ]; then
    cp -a /root/.windsurf-server/* /workspace/.persist/windsurf-server/ 2>/dev/null || true
fi
if [ -d /root/.codeium ] && [ ! -L /root/.codeium ]; then
    cp -a /root/.codeium/* /workspace/.codeium_persist/ 2>/dev/null || true
fi

rm -rf /root/.windsurf-server
ln -s /workspace/.persist/windsurf-server /root/.windsurf-server

rm -rf /root/.codeium
ln -s /workspace/.codeium_persist /root/.codeium

echo "✓ Windsurf persistence restored"
