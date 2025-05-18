from etl_architect_agent_v2.backend.config import get_settings

settings = get_settings()
print(f"AWS Region: {settings.AWS_REGION}")
print(f"S3 Bucket: {settings.AWS_S3_BUCKET}")