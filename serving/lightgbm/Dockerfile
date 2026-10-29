FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py .

EXPOSE 8080

CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8080"]
