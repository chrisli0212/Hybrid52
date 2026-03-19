ash
# Kill everything related to your project
pkill -f run_all.py
pkill -f prediction_service.py
pkill -f theta_dashboard_dash.py
pkill -f theta_fetching_v5.py

# Also free port 8050 if still held
fuser -k 8050/tcp
Or the nuclear option to kill all Python processes at once:

bash
pkill -f python
Then restart fresh:

bash
source /workspace/venv/bin/activate
cd /workspace/Final_production_model
python run_all.py