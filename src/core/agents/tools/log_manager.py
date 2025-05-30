import os
import json
import boto3
from datetime import datetime, timedelta
from threading import Lock
from typing import List


class LogManager:
    def __init__(self, bucket, region, log_dir='logs', retention_minutes=10):
        self.bucket = bucket
        self.region = region
        self.log_dir = log_dir
        self.retention_minutes = retention_minutes
        self.s3 = boto3.client('s3', region_name=region)
        self.lock = Lock()
        os.makedirs(log_dir, exist_ok=True)

    def get_log_file_path(self):
        now = datetime.utcnow()
        return os.path.join(
            self.log_dir,
            f"tool_logs_{now.strftime('%Y%m%d_%H%M')}.log"
        )

    def write_log(self, log_entry):
        path = self.get_log_file_path()
        with self.lock:
            with open(path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    def upload_logs_to_s3(self):
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=self.retention_minutes)
        for fname in os.listdir(self.log_dir):
            if fname.startswith("tool_logs_") and fname.endswith(".log"):
                timestamp_str = fname[len("tool_logs_"):-4]
                try:
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                except Exception:
                    continue
                local_path = os.path.join(self.log_dir, fname)
                s3_key = f"logs/{fname}"
                if file_time >= cutoff:
                    # Upload recent logs
                    try:
                        self.s3.upload_file(local_path, self.bucket, s3_key)
                    except Exception as e:
                        print(f"Failed to upload {fname} to S3: {e}")
                else:
                    # Remove old logs locally and from S3
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
                    try:
                        self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
                    except Exception:
                        pass

    def list_recent_logs(self) -> List[str]:
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=self.retention_minutes)
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix="logs/")
        logs = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".log"):
                timestamp_str = key.split("_")[-1][:-4]
                try:
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                except Exception:
                    continue
                if file_time >= cutoff:
                    logs.append(key)
        return sorted(logs) 