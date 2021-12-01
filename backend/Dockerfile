FROM python:3.8

# Set working directory
RUN mkdir -p /usr/src/backend
WORKDIR /usr/src/backend
ADD src/requirements.txt /usr/src/backend/src/requirements.txt

# Install required packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r ./src/requirements.txt

# Start Flask app along with .env settings
ENTRYPOINT ./src/bootstrap.sh

