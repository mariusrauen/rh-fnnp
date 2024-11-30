# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /workspace/

# Install system dependencies
RUN apt-get update && apt-get install -y curl && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy the `pyproject.toml` and `poetry.lock` files
COPY pyproject.toml poetry.lock* /workspace/

# Install dependencies with Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application code
COPY . /workspace/

# Command to keep the container running
CMD ["bash"]