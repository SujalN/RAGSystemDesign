FROM python:3.13-slim-alpine

# Set working directory
WORKDIR /app

# Copy only requirements first
COPY ../requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy UI code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]