FROM python:3.13-slim-alpine

# Set working directory
WORKDIR /app

# Copy only requirements first for cache efficiency
COPY ../requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the backend app
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]