Dockerizing Instructions

`docker build -t cdaprod/cda-minio-control .`

```bash
docker run -p 8000:8000 --name cda-minio-control \
  -e OPENAI_API_KEY=<your-openai-api-key> \
  -e MINIO_ENDPOINT=<your-minio-endpoint> \
  -e MINIO_ACCESS_KEY=<your-minio-access-key> \
  -e MINIO_SECRET_KEY=<your-minio-secret-key> \
  cdaprod/cda-minio-control
```

`docker push cdaprod/cda-minio-control`

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Login to GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.repository_owner }}
        password: ${{ secrets.GH_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: |
          cdaprod/cda-minio-control:latest
          ghcr.io/cdaprod/cda-minio-control:latest
``` 