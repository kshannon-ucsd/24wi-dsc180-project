# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy RF model API source files
COPY src/app.py src/model_loader.py /app/

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the container port
EXPOSE 8080

# Run Flask using Gunicorn (better for production)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
