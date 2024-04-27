# Base image
FROM pytorch/pytorch

LABEL org.opencontainers.image.source = "https://github.com/mamdollah/cross-domain-image-feature-extraction"

# Install OpenGL library
RUN apt-get update && apt-get install -y \ 
	libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        && rm -rf /var/lib/apt/lists/*

# working directory in the container
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of project code into the container
COPY . /app

CMD ["python3", "experiments/atari/breakout/blocks/parallel_runs.py"]