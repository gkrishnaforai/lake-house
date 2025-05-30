import os
import time
import boto3
import pytest
import requests

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
TABLE_NAME = "integration_test_table"
FILE_PATH = "/Users/krishnag/Documents/fromWalter/Sample Data for Krishna to Train AI on Y-N sorting task 5-7-25 small.xlsx"
DYNAMODB_TABLE = "etl_workflow_states"
S3_BUCKET = "lambda-code-q"
  # <-- Set to your actual bucket
GLUE_DATABASE = "user_test_user"            # <-- Set to your actual Glue DB

def test_etl_workflow_end_to_end():
    # 1. Upload file to API
    with open(FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(FILE_PATH), f)}
        resp = requests.post(
            f"{API_URL}/api/catalog/tables/{TABLE_NAME}/files",
            files=files,
            data={"database_name": GLUE_DATABASE}
        )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "success"
    workflow_id = data["workflow_id"]

    # 2. Poll DynamoDB for workflow status
    dynamodb = boto3.client("dynamodb")
    for _ in range(30):  # Wait up to 150 seconds
        item = dynamodb.get_item(
            TableName=DYNAMODB_TABLE,
            Key={"workflow_id": {"S": workflow_id}}
        ).get("Item")
        assert item, f"Workflow {workflow_id} not found in DynamoDB"
        status = item["status"]["S"]
        print(f"Workflow status: {status}")
        if status == "COMPLETED":
            break
        elif status == "FAILED":
            pytest.fail(f"Workflow failed: {item}")
        time.sleep(5)
    else:
        pytest.fail("Workflow did not complete in time")

    # 3. Validate file in S3
    s3 = boto3.client("s3")
    prefix = f"tables/{TABLE_NAME}"
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    assert "Contents" in resp and len(resp["Contents"]) > 0, "File not found in S3"
    print(f"S3 objects under {prefix}: {[obj['Key'] for obj in resp['Contents']]}")

    # 4. Validate table in Glue
    glue = boto3.client("glue")
    table = glue.get_table(DatabaseName=GLUE_DATABASE, Name=TABLE_NAME)["Table"]
    assert table["Name"] == TABLE_NAME
    assert table["TableType"] == "ICEBERG"
    print(f"Glue table columns: {[col['Name'] for col in table['StorageDescriptor']['Columns']]}")

    # 5. Validate workflow results
    results = item["results"]["M"]
    assert "schema_inference" in results
    assert "file_loading" in results
    assert "table_validation" in results
    print("Workflow results:", results)

    print("Integration test PASSED.")

if __name__ == "__main__":
    test_etl_workflow_end_to_end() 