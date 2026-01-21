# 1. Base Image: Use a lightweight version of Python 3.12 (Linux based)
FROM python:3.12-slim

# 2. Set the working directory inside the container to /app
#    (All future commands run from here)
WORKDIR /app

# 3. Copy just the requirements first (Optimization technique)
#    Docker caches layers. If requirements.txt doesn't change, 
#    it won't re-run the slow 'pip install' step.
COPY ./requirements.txt .

# 4. Install dependencies
#    --no-cache-dir keeps the image smaller by removing download caches
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your source code into the container
COPY . .

# 6. Expose the port Streamlit runs on (default 8501)
EXPOSE 8501

# 7. The command to start the app
#    We use the path './app.py' because we copied it to /app/
#    address=0.0.0.0 is REQUIRED for Docker to share the port with outside world
CMD ["streamlit", "run", "./app.py", "--server.address=0.0.0.0"]