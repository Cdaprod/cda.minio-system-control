Certainly! Let's containerize the application and create a GitHub Actions workflow to build and push the Docker image to both GitHub Container Registry (GHCR) and Docker Hub.

Here's a step-by-step guide:

1. Create a Dockerfile in your project's root directory:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

2. Create a `.dockerignore` file to exclude unnecessary files from the Docker image:

```
.git
.gitignore
Dockerfile
.dockerignore
README.md
```

3. Create a GitHub Actions workflow file (e.g., `.github/workflows/docker-build-push.yml`) with the following content:

```yaml
name: Docker Build and Push

on:
  push:
    branches:
      - main

env:
  DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
  DOCKER_PASSWORD: ${{ secrets.DOCKER_PASSWORD }}
  GHCR_USERNAME: ${{ github.actor }}
  GHCR_TOKEN: ${{ secrets.GITHUB_TOKEN }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Build Docker image
      run: |
        docker build -t minio-langchain-app .

    - name: Login to Docker Hub
      run: |
        echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin

    - name: Push to Docker Hub
      run: |
        docker tag minio-langchain-app ${{ secrets.DOCKER_USERNAME }}/minio-langchain-app:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/minio-langchain-app:latest

    - name: Login to GitHub Container Registry
      run: |
        echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u "${{ github.actor }}" --password-stdin

    - name: Push to GitHub Container Registry
      run: |
        docker tag minio-langchain-app ghcr.io/${{ github.repository_owner }}/minio-langchain-app:latest
        docker push ghcr.io/${{ github.repository_owner }}/minio-langchain-app:latest
```

4. Set up the required secrets in your GitHub repository:
   - Go to your repository's "Settings" > "Secrets" > "Actions".
   - Add the following secrets:
     - `DOCKER_USERNAME`: Your Docker Hub username.
     - `DOCKER_PASSWORD`: Your Docker Hub password or access token.

5. Commit and push the Dockerfile, .dockerignore, and the GitHub Actions workflow file to your repository.

6. The GitHub Actions workflow will be triggered automatically when you push changes to the `main` branch. It will build the Docker image and push it to both Docker Hub and GitHub Container Registry (GHCR).

7. After the workflow completes successfully, you can find the Docker image in your Docker Hub repository and GHCR repository.

To run the containerized application, use the following command:

```
docker run -it --rm minio-langchain-app
```

Make sure to replace `minio-langchain-app` with the actual image name you used in the Dockerfile and GitHub Actions workflow.

That's it! You have now containerized your application and set up a GitHub Actions workflow to automatically build and push the Docker image to both Docker Hub and GHCR whenever changes are pushed to the `main` branch.