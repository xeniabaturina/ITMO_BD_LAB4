FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Ansible and required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ansible \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install pyyaml

# Copy config.ini first
COPY config.ini /app/config.ini

# Copy Ansible files
COPY ansible /app/ansible/

# Encrypt the secrets file during build
RUN ansible-vault encrypt /app/ansible/secrets.yml --vault-password-file /app/ansible/vault_password.txt

# Copy the rest of the application
COPY . /app/

RUN mkdir -p logs results experiments data

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

# Simple direct command to run the API
CMD ["python", "-m", "src.api"]
