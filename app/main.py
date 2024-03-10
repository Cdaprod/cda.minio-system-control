import asyncio
import os
import re
import io
import tempfile
import uuid
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from minio import Minio
import requests
from unstructured.partition.auto import partition
from docker import from_env as docker_from_env
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
import uvicorn

from minio_langchain_tools import upload_file_to_minio, download_file_from_minio, list_objects_in_minio_bucket
from hydrate_integration import hydrate_data
from langchain_integration import execute_langchain_query
from agent_control_integration import execute_agent_action

class MinIOConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = True

class DockerConfig(BaseModel):
    image: str
    command: str
    volumes: dict
    detach: bool = True
    remove: bool = True

class URLItem(BaseModel):
    url: HttpUrl

class Document(BaseModel):
    source: str
    content: str

class ScriptExecutionRequest(BaseModel):
    bucket_name: str
    script_name: str

class LangChainInput(BaseModel):
    input_text: str

def sanitize_url_to_object_name(url: str) -> str:
    clean_url = re.sub(r'^https?://', '', url)
    clean_url = re.sub(r'[^\w\-_\.]', '_', clean_url)
    return clean_url[:250] + '.txt'

def prepare_text_for_tokenization(text: str) -> str:
    clean_text = re.sub(r'\s+', ' ', text).strip()
    return clean_text

class MinIOOperations:
    def __init__(self, config: MinIOConfig):
        self.client = Minio(config.endpoint, access_key=config.access_key, secret_key=config.secret_key, secure=config.secure)

    async def store_object(self, bucket_name: str, object_name: str, file_path: str):
        self.client.fput_object(bucket_name, object_name, file_path)
    
    async def retrieve_object(self, bucket_name: str, object_name: str, file_path: str):
        self.client.fget_object(bucket_name, object_name, file_path)

class DockerOperations:
    def __init__(self, config: DockerConfig):
        self.client = docker_from_env()
        self.config = config

    async def execute_script(self, script_content: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as script_file:
            script_file.write(script_content.encode())
            script_path = script_file.name
        container_name = f"script_execution_{uuid.uuid4()}"
        container = self.client.containers.run(
            image=self.config.image,
            command=f"python {script_path}",
            volumes=self.config.volumes,
            detach=self.config.detach,
            remove=self.config.remove,
            name=container_name
        )
        return container_name

class MinIOSystemOrchestrator:
    def __init__(self, minio_config: MinIOConfig, docker_config: DockerConfig):
        self.minio_ops = MinIOOperations(minio_config)
        self.docker_ops = DockerOperations(docker_config)
    
    async def execute_url_etl_pipeline(self, urls: List[URLItem]):
        async with ThreadPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(executor, self.process_url, url) for url in urls]
            await asyncio.gather(*tasks)
    
    def process_url(self, url_item: URLItem):
        response = requests.get(url_item.url)
        response.raise_for_status()
        html_content = io.BytesIO(response.content)
        elements = partition(file=html_content, content_type="text/html")
        combined_text = "\n".join([e.text for e in elements if hasattr(e, 'text')])
        combined_text = prepare_text_for_tokenization(combined_text)
        object_name = sanitize_url_to_object_name(url_item.url)
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as tmp_file:
            tmp_file.write(combined_text)
            tmp_file_path = tmp_file.name
        asyncio.run(self.minio_ops.store_object("your_bucket_name", object_name, tmp_file_path))
        os.remove(tmp_file_path)

    async def execute_script_in_docker(self, execution_request: ScriptExecutionRequest):
        response = self.minio_ops.client.get_object(execution_request.bucket_name, execution_request.script_name)
        script_content = response.read().decode('utf-8')
        container_name = await self.docker_ops.execute_script(script_content)
        return {"message": "Execution started", "container_name": container_name}

app = FastAPI()

@app.post("/execute/{bucket_name}/{script_name}")
async def execute_script(bucket_name: str, script_name: str):
    orchestrator = MinIOSystemOrchestrator(
        MinIOConfig(endpoint="play.min.io:443", access_key="minioadmin", secret_key="minioadmin"),
        DockerConfig(image="python:3.8-slim", command="", volumes={}, detach=True, remove=True)
    )
    container_name = await orchestrator.execute_script_in_docker(ScriptExecutionRequest(bucket_name=bucket_name, script_name=script_name))
    return {"message": "Execution started", "container_name": container_name}

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/langchain-execute/")
async def execute_langchain_integration(input: LangChainInput):
    result = await execute_langchain_query(input.input_text)
    return {"message": "LangChain processing completed", "result": result}

@app.post("/hydrate-data/")
async def hydrate_data_endpoint(input: dict):
    result = await hydrate_data(input)
    return {"message": "Data hydration completed", "result": result}

@app.post("/agent-action/")
async def execute_agent_action_endpoint(input: dict):
    result = await execute_agent_action(input)
    return {"message": "Agent action executed", "result": result}

@app.post("/minio-webhook/")
async def handle_minio_webhook(event: dict):
    event_name = event.get("EventName")
    if event_name == "s3:ObjectCreated:Put":
        pass
    elif event_name == "s3:ObjectRemoved:Delete":
        pass
    return {"message": "Webhook event processed successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)