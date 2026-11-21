FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
# train the artifact at build time so the image is self-contained
RUN PYTHONPATH=src python scripts/run_experiments.py > /dev/null
ENV PYTHONPATH=src PRICERAG_MODEL=artifacts/pricerag.joblib
EXPOSE 8000
CMD ["uvicorn", "pricerag.api:app", "--host", "0.0.0.0", "--port", "8000"]
