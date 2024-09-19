
FROM python:3.10-bookworm

RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    procps \
    git-lfs \
    zip \
    unzip \
    htop \
    vim \
    nano \
    bzip2 \
    libx11-6 \
    build-essential \
    libsndfile-dev \
    software-properties-common \
    libxrender1 \
    xsel \
    xclip \
 && rm -rf /var/lib/apt/lists/*


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

RUN pip install --upgrade pip setuptools wheel

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache


COPY --chown=+1000 . $HOME/app

#RUN mkdir -p /data 
#RUN chmod 777 /data # set permissions for additional files

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR
#RUN poetry add huggingface_hub ############### for hf container
# Might change this later but for now I'd like to have the apt-get installed packages available inside the container
# The runtime image, used to just run the code provided its virtual environment
#FROM python:3.10-slim-buster as runtime

# Put the python venv bin at the beginning of PATH to set it as default cmd pythion
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="$HOME/app/.venv/bin/python" \
    DOCKER_INI=true

#COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
#EXPOSE 7860
#CMD ["python", "/app.py"] ############### for hf container
CMD ["/bin/bash"]
#ENTRYPOINT [".venv/bin/python3", "./cm_retro/run_excel_pipeline.py"] 
#"./cm_retro/data/input/input.xlsx"
