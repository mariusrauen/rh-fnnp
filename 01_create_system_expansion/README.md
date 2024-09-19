# Run with Docker
- install docker (https://docs.docker.com/desktop/install/windows-install/) and start it
- build with `docker build . --tag=system_expansion.latest`
- run with `docker run -it --rm -v "$PWD/data:/data" system_expansion.latest`
- find your files from ./data in /data inside the container
- run the scripts with "python <script>" as usual