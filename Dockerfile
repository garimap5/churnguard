FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
RUN PYTHONPATH=src python scripts/run_all.py > /dev/null
ENV PYTHONPATH=src CHURNGUARD_BUNDLE=artifacts/churnguard.joblib
EXPOSE 8000
CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
