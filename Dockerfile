# Use an official lightweight Python image
FROM python:3.9-slim

# Set environment variables for better behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=120", "--preload", "--bind", "0.0.0.0:8000", "app:server"]
