import boto3
import sys
import json


def get_caller_identity():
    """Get information about the current IAM identity."""
    sts = boto3.client('sts')
    try:
        response = sts.get_caller_identity()
        return response
    except Exception as e:
        print(f"Error getting caller identity: {str(e)}")
        return None


def check_s3_access(bucket: str, key: str):
    """Check S3 access permissions."""
    s3 = boto3.client('s3')
    try:
        # Try to get object metadata
        response = s3.head_object(Bucket=bucket, Key=key)
        print(f"\nS3 Access Check:")
        print(f"  Can access file: Yes")
        print(f"  File size: {response.get('ContentLength', 0)} bytes")
        print(f"  Last modified: {response.get('LastModified')}")
        return True
    except Exception as e:
        print(f"\nS3 Access Check:")
        print(f"  Can access file: No")
        print(f"  Error: {str(e)}")
        return False


def check_athena_permissions():
    """Check Athena permissions."""
    athena = boto3.client('athena')
    try:
        # Try to list workgroups
        response = athena.list_work_groups()
        print(f"\nAthena Permissions Check:")
        print(f"  Can list workgroups: Yes")
        print(f"  Available workgroups: {[wg['Name'] for wg in response.get('WorkGroups', [])]}")
        return True
    except Exception as e:
        print(f"\nAthena Permissions Check:")
        print(f"  Can list workgroups: No")
        print(f"  Error: {str(e)}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_iam_permissions.py <s3_path>")
        print("Example: python check_iam_permissions.py s3://bucket/path/to/file.parquet")
        sys.exit(1)
        
    s3_path = sys.argv[1]
    
    # Parse S3 path
    if not s3_path.startswith('s3://'):
        print("Error: Invalid S3 path. Must start with 's3://'")
        sys.exit(1)
        
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    
    # 1. Check IAM identity
    print("\nIAM Identity Check:")
    print("-" * 50)
    identity = get_caller_identity()
    if identity:
        print(f"Account: {identity['Account']}")
        print(f"User ID: {identity['UserId']}")
        print(f"ARN: {identity['Arn']}")
    
    # 2. Check S3 access
    check_s3_access(bucket, key)
    
    # 3. Check Athena permissions
    check_athena_permissions()


if __name__ == "__main__":
    main() 