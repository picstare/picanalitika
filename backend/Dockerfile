# Use the official Python base image
FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# Set the working directory in the container
WORKDIR /backend/

# Copy the Django project files to the container
COPY backend /backend/

# Install build dependencies and required packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libc6-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove \
        build-essential \
        gcc \
        libc6-dev \
        linux-headers-amd64 \
    && rm -rf /var/lib/apt/lists/*



# Expose the Django development server port
EXPOSE 8000

# Command to run Django development server
CMD python manage.py runserver 0.0.0.0:8000
