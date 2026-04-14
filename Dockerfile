FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download data at build time (optional — can also be done at startup)
RUN python -c "from data.data_loader import MovieLensLoader; MovieLensLoader()" || true

EXPOSE 8000 8501

# Default: run API. Override CMD to run Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
