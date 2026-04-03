# Use slim base
FROM python:3.11-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies (no cache = smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of app
COPY . .

# Correct port (Streamlit default)
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "myapp.py", "--server.port=8501", "--server.address=0.0.0.0"]



# # Use a Python 3.11 base image to match your local environment
# FROM python:3.11-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the requirements file and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy your app code and the model folder
# COPY . .

# # Streamlit runs on port 8501 by default
# EXPOSE 8501

# # Command to run the app
# CMD ["streamlit", "run", "myapp.py", "--server.port=8501", "--server.address=0.0.0.0"]