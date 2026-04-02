#!/usr/bin/env python3
# Download runpod-backup/historical_data_1yr/ -> /workspace/historical_data_1yr/

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import boto3
    from botocore.client import Config
except ModuleNotFoundError:
    sys.stderr.write("Install boto3: pip install boto3\n")
    sys.exit(1)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

# ===== Load .env (same pattern as your existing scripts) =====
def _load_r2_env():
    for base in (Path("/workspace/hybrid46"), Path("/workspace"), Path(".")):
        env_file = base / ".env"
        if env_file.is_file():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            break

_load_r2_env()

R2_ACCOUNT_ID     = os.environ.get("R2_ACCOUNT_ID", "").strip()
R2_ACCESS_KEY_ID  = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()

BUCKET   = "runpod-backup"
PREFIX   = "runpod-backup/historical_data_1yr/"
LOCAL_DEST = "/workspace"
WORKERS  = 8

_print_lock = threading.Lock()

def get_s3_client():
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        raise SystemExit("R2 credentials missing. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in .env")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
        verify=False,
    )

def format_size(b):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024: return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"

def list_objects(s3, bucket, prefix):
    files = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token: kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"] != prefix.rstrip("/"):
                files.append({"key": obj["Key"], "size": obj.get("Size", 0)})
        if not resp.get("IsTruncated"): break
        token = resp.get("NextContinuationToken")
    return files

def download_one(args):
    s3, bucket, key, local_path = args
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            with _print_lock: print(f"  [SKIP] {os.path.basename(local_path)}")
            return True
        s3.download_file(bucket, key, local_path)
        with _print_lock: print(f"  [OK] {os.path.basename(local_path)} ({format_size(os.path.getsize(local_path))})")
        return True
    except Exception as e:
        with _print_lock: print(f"  [ERR] {key}: {e}")
        return False

if __name__ == "__main__":
    s3 = get_s3_client()
    print(f"Bucket : {BUCKET}")
    print(f"Prefix : {PREFIX}")
    print(f"Dest   : {LOCAL_DEST}")
    print("Testing connection...")
    try:
        s3.head_bucket(Bucket=BUCKET)
        print("✓ Connected\n")
    except Exception as e:
        print(f"✗ Failed: {e}"); sys.exit(1)

    files = list_objects(s3, BUCKET, PREFIX)
    if not files:
        print("No files found."); sys.exit(0)

    total_size = sum(f["size"] for f in files)
    print(f"Found {len(files)} file(s), {format_size(total_size)}\n")

    # Strip the R2 prefix, download flat into /workspace/historical_data_1yr/
    tasks = []
    for f in files:
        rel = f["key"][len(PREFIX):]           # strip leading prefix
        local_path = os.path.join(LOCAL_DEST, "historical_data_1yr", rel)
        tasks.append((s3, BUCKET, f["key"], local_path))

    ok = fail = done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_one, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  Progress: {done}/{len(tasks)}")
            if fut.result(): ok += 1
            else: fail += 1

    print(f"\nDone: {ok} OK / {fail} failed")
    sys.exit(1 if fail else 0)