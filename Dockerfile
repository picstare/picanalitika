FROM python:3.9

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel
RUN pip install nltk

RUN pip install -r requirements.txt
COPY nltk.txt /app/nltk.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "Home.py"]
ENTRYPOINT ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableWebsocketCompression=false", "Home.py"]



RUN python -c "import nltk; nltk.download(open('/app/nltk.txt').read().splitlines())"





# app/Dockerfile
# FROM python:3.9 As requirements

# WORKDIR /picanalitika

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY . .

# # RUN git clone https://github.com/streamlit/streamlit-example.git .


# RUN pip install -r requirements.txt --no-deps

# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Use the official Python base image
# FROM python:3.9 AS baseImage

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the requirements file to the working directory
# COPY . .

# # Install the required dependencies
# RUN pip install --no-cache-dir -r requirements.txt --no-deps


# # Expose the ports for Django admin and Streamlit
# EXPOSE 8000
# ENTRYPOINT  ["django", "manage.py", "runser", "--server.port=8000", "--server.address=127.0.0.1"]

# FROM baseImage

# EXPOSE 8501

# # Start Gunicorn to serve the Django application

# ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=127.0.0.1"]
# app/Dockerfile
# FROM python:3.9 AS baseImage

# # Set the working directory inside the container
# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy the requirements file to the working directory
# RUN git clone https://github.com/picstare/picanalytica.git .

# # Install the required dependencies
# RUN pip install --no-cache-dir -r requirements.txt --no-deps

# # Expose the ports for Django admin and Streamlit
# EXPOSE 8000
# EXPOSE 8501

# # Set the entrypoint script
# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh
# CMD ["/entrypoint.sh"]

# Use the official Python base image
# FROM python:3.9

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# # Set the working directory in the container
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     vim \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# COPY requirements.txt /app/
# RUN pip install --upgrade pip && pip install -r requirements.txt

# # Copy the entire project folder to the container
# COPY . /app/

# # Expose the port that Streamlit will run on
# EXPOSE 8501

# # Command to run Streamlit as the main landing site
# CMD streamlit run Home.py