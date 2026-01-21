# stop all containers and delete all volumes
# named volumes & anonymous volume directories are automatically removed from server or local machine
docker compose down --volumes

# remove bind directories from server or local machine
sudo rm -rf ./chroma_data