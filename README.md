# corrective-rag

### Steps

- install deps in venv
- run fillDb.py with a context file of your choice
- run corrective-rag: give it a question and it should answer

### Reminders

##### Image vs Container

- 1 container = 1 image
- image (immutable): contains filesystem (os+libraries+binaries), your code, your dependencies, docker config/metadata (WORKDIR, EXPOSE, ENV vars, CMD/ENTRYPOINT)
- container: creates a writable layer on top of an image to run processes (defined by CMD/ENTRYPOINT in image, or when you run "docker exec" yourself)

- container can override certain image config like env var (via docker-compose)
- container exposes ports as the image tells it to do

##### Dockerfile vs docker-compose.yml

- dockerfile: create image using your own code
- docker-compose.yml: run 1/+ containers, each with its image, volumes, ports, env vars, etc. set up

- dockerfile command COPY copies from your local folders to WORKDIR of the container
- dockerfile command RUN runs the command in WORKDIR of the container

##### why use Volumes?

- Q: why use volumes when you can just set up the db in your server and connect docker code to it via db url as env var?
- A: if you want to change server, you have to manually set up db again. you also pollute the server with db configs
- A: in contrast, if you use volumes, when you change server you still just run 1 cmd. Also servers dont need to store db configs.
- Q: then why not put the db in the image?
- A: too big.

##### Bind vs Named volumes

- Both volumes are stored in the machine, not the image.
- Bind: choose an arbitrary server directory (docker will set up db there) and map it to an image directory (then your image code can access that image dir for data).
- You can view/edit the files in the chosen directory.
- Named: choose an arbitrary name and map it to an image directory. Also in docker-compose.yml, in addition to mapping, you need to declare all named volumes at the top level once.
- Named volumes are stored in /var/lib/docker/volumes. You can also view/edit files there, but it's not recommended.
