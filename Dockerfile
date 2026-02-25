FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# This tells Python to look in the current folder for your modules
ENV PYTHONPATH=/app

RUN addgroup --system bio && adduser --system --group bio

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=bio:bio . .

USER bio

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
