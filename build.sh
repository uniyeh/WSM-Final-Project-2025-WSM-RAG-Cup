#!/bin/bash

docker compose up -d

echo "Pulling models... (This may take a while)"
docker compose exec ollama ollama pull granite4:3b
docker compose exec ollama ollama pull embeddinggemma:300m
# docker compose exec ollama ollama pull qwen3-embedding:0.6b
# docker compose exec ollama ollama pull BAAI/bge-reranker-v2-m3
echo "Models pulled! Server is ready."

echo "Enter the container to run the system"
docker compose exec rag-app bash