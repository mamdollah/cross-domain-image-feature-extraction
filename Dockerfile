# Base image
FROM pytorch/pytorch:latest

LABEL org.opencontainers.image.source = "https://github.com/mamdollah/cross-domain-image-feature-extraction"

# working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
# Copy the rest of project code into the container
COPY experiments .
COPY feature_extraction .
COPY utils.py .

COPY .github .
COPY .gitignore .
COPY Dockerfile .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Install additional packages for OpenAI Gym (assuming you're using it for RL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev
