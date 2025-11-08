FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev pkg-config \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your app code
COPY . .

# Expose port and start server
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
