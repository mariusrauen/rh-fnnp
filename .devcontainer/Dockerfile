FROM python:3.10-bookworm

# Install system dependencies
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
RUN useradd -m -u 1000 user \
    && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV VIRTUAL_ENV=/home/user/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["/bin/bash"]