FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (needed for compilation sometimes)
RUN apt-get update && apt-get install -y gcc

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
COPY backend/ ./backend/
COPY model/ ./model/

# Create a volume mount point for persistence
VOLUME /app/model

# Expose Port
EXPOSE 8000

# Run
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
