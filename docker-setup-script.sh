# start all containers & create volume directories
# -d means "detached" (run in background)
docker compose up -d --build

# one time: fill ollama volumes (with models)
# Execute a command 'exec' inside the running 'ollama_backend' container
docker exec -it ollama_backend ollama pull llama3.2

# one time: fill vector db volumes (with data)
# Execute the fill-vector-db script inside the 'rag_app' container
docker exec -it rag_app python fill-vector-db.py