# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (ignoring cache to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app runs on (app.py uses 8000)
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]