# Use a lightweight Python image (optimized for containers)
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all necessary application files
COPY src/app.py src/model_loader.py /app/

# Expose the container port for the Flask API
EXPOSE 8080

# Run the Flask app using Gunicorn (better for production)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
