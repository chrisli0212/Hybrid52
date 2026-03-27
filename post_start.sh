#!/bin/bash
echo "→ Starting post-start setup..."

# === Jupyter ===
echo "→ Setting up Jupyter..."
/usr/bin/python3.11 -m pip install jupyterlab -q
pkill -f "python3.10.*jupyter" 2>/dev/null || true
nohup /usr/bin/python3.11 -m jupyter lab \
  --allow-root --no-browser --port=8888 --ip=* \
  --FileContentsManager.delete_to_trash=False \
  --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
  --ServerApp.token='' --ServerApp.password='' \
  --ServerApp.allow_origin=* \
  --ServerApp.preferred_dir=/workspace \
  &> /jupyter.log &

# === Editor + AI Memory Persistence (Cursor + Windsurf/Verdent) ===
echo "→ Setting up persistent AI memory..."

# Create folders on the persistent /workspace volume
mkdir -p /workspace/.config/Cursor \
         /workspace/.cursor \
         /workspace/.persist/codeium \
         /workspace/.persist/windsurf-server \
         /workspace/.verdent \
         /workspace/.vscode-server \
         /workspace/.persist/cursor-config/User/workspaceStorage

# Symlink everything so AI memory survives pod changes/restarts
rm -rf /root/.config/Cursor 2>/dev/null || true
ln -sfn /workspace/.persist/cursor-config /root/.config/Cursor

rm -rf /root/.cursor 2>/dev/null || true
ln -sfn /workspace/.cursor /root/.cursor

rm -rf /root/.verdent 2>/dev/null || true
ln -sfn /workspace/.verdent /root/.verdent

rm -rf /root/.vscode-server 2>/dev/null || true
ln -sfn /workspace/.vscode-server /root/.vscode-server

echo "→ Restoring Cursor & Windsurf..."
bash /workspace/.windsurf_persist/setup_symlinks.sh || echo "Warning: Windsurf symlinks setup failed"
bash /workspace/init_windsurf.sh || echo "Warning: Windsurf init failed"
bash /workspace/init_cursor.sh || echo "Warning: Cursor init failed"

echo "✅ Cursor + Windsurf/Verdent AI memory now 100% persisted!"

# === VENV ACTIVATION (now bulletproof) ===
echo "→ Setting up venv..."
mkdir -p /workspace/venv
grep -q 'workspace/venv' /root/.bashrc 2>/dev/null || echo 'source /workspace/venv/bin/activate' >> /root/.bashrc
# Force activation for ALL shells (including Cursor subprocesses)
export VIRTUAL_ENV=/workspace/venv
export PATH="/workspace/venv/bin:$PATH"
export PYTHONPATH=/workspace/venv/lib/python*/site-packages:$PYTHONPATH 2>/dev/null || true
source /workspace/venv/bin/activate 2>/dev/null || true

if [ -f /workspace/requirements.txt ]; then
    echo "→ Restoring packages from requirements.txt..."
    bash /workspace/pip-install.sh
else
    echo "→ No requirements.txt yet — skipping"
fi

# Jupyter kernel (now uses venv)
python -c "import ipykernel" 2>/dev/null || pip install ipykernel -q
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
echo "✓ Post-start complete (Jupyter at port 8888)"

# === Auto-run Final Production Model (run_all.py) ===
echo "→ Starting Final Production Model..."

# Clean up old instance (prevents duplicates on every restart)
if [ -f /workspace/run_all.pid ]; then
    kill $(cat /workspace/run_all.pid 2>/dev/null) 2>/dev/null || true
    rm -f /workspace/run_all.pid
fi

cd /workspace/Final_production_model || { echo "⚠️ Directory /workspace/Final_production_model not found!"; }

source /workspace/venv/bin/activate

nohup python run_all.py > /workspace/run_all.log 2>&1 &
echo $! > /workspace/run_all.pid

echo "✅ Production model (run_all.py) started in background"
echo "   Log: /workspace/run_all.log"
echo "   PID: $(cat /workspace/run_all.pid 2>/dev/null)"
