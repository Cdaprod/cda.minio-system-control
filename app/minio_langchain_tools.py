from llama_index.tools import FunctionTool
from minio import Minio
from minio.error import S3Error

# Initialize Minio client
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "play.min.io"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=True
)

# Function to create a bucket
def create_bucket(bucket_name):
    try:
        minio_client.make_bucket(bucket_name)
        return f"Bucket {bucket_name} created successfully."
    except S3Error as e:
        return f"Failed to create bucket {bucket_name}: {str(e)}"

# Function to upload a file
def upload_file(bucket_name, file_path, object_name):
    try:
        minio_client.fput_object(bucket_name, object_name, file_path)
        return f"File {file_path} uploaded as {object_name} in bucket {bucket_name}."
    except S3Error as e:
        return f"Failed to upload file {file_path} to bucket {bucket_name}: {str(e)}"

# Function to list buckets
def list_buckets():
    try:
        buckets = minio_client.list_buckets()
        bucket_list = [bucket.name for bucket in buckets]
        return f"Buckets: {', '.join(bucket_list)}"
    except S3Error as e:
        return f"Failed to list buckets: {str(e)}"

# Function to delete a file
def delete_file(bucket_name, object_name):
    try:
        minio_client.remove_object(bucket_name, object_name)
        return f"File {object_name} deleted from bucket {bucket_name}."
    except S3Error as e:
        return f"Failed to delete file {object_name} from bucket {bucket_name}: {str(e)}"

# Creating FunctionTool objects for each operation
create_bucket_tool = FunctionTool.from_defaults(
    fn=create_bucket,
    name="create_bucket",
    description="Creates a new bucket in Minio"
)

upload_file_tool = FunctionTool.from_defaults(
    fn=upload_file,
    name="upload_file",
    description="Uploads a file to a specified bucket in Minio"
)

list_buckets_tool = FunctionTool.from_defaults(
    fn=list_buckets,
    name="list_buckets",
    description="Lists all buckets in Minio"
)

delete_file_tool = FunctionTool.from_defaults(
    fn=delete_file,
    name="delete_file",
    description="Deletes a specified file from a bucket in Minio"
)

# Example: Saving the tools in a list or dict for access
minio_tools = [create_bucket_tool, upload_file_tool, list_buckets_tool, delete_file_tool]

# You can access and call any tool from minio_tools based on your application's logic.