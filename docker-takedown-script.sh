# stop all containers the current docker-compose.yml has and delete all volumes
# named volumes & anonymous volume directories are automatically removed from server or local machine
docker compose down --volumes

# remove bind directories from server or local machine
sudo rm -rf ./chroma_data

# remove images
docker rmi wenboliu68/ollama-rag:latest ollama/ollama:latest