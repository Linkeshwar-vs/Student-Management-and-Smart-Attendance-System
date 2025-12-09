# Use slim Python 3.8 image
FROM python:3.8.18-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install everything except TensorFlow stack
RUN pip install --upgrade pip \
    && pip --default-timeout=2000 install --no-cache-dir -r requirements.txt


# Force install TensorFlow stack with no-deps
RUN pip install --no-deps tensorflow==2.13.0 \
    && pip install --no-deps tensorflow-estimator==2.13.0 \
    && pip install --no-deps keras==2.13.1 \
    && pip install --no-deps tensorboard==2.13.0

# Force install albumentations and albucore without deps
RUN pip install --no-deps albucore==0.0.17 \
    && pip install --no-deps albumentations==1.4.18

RUN pip install --no-deps typing_extensions==4.11.0
# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
