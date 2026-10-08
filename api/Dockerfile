# Product search API. Build from the repo root:
#   docker build -f api/Dockerfile -t product-search-api .
# Layers go from least to most often changed, so a code edit only rebuilds the last one.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TRANSFORMERS_VERBOSITY=error \
    TQDM_DISABLE=1

WORKDIR /app

# 1. Dependencies (CPU-only torch keeps the image ~2 GB smaller than the CUDA build).
COPY api/requirements.txt api/requirements.txt
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r api/requirements.txt

# 2. Models (~1.6 GB), baked in so a cold start never downloads from Hugging Face.
COPY api/download_models.py api/download_models.py
RUN python api/download_models.py
ENV HF_HUB_OFFLINE=1

# 3. The read-only index (python api/export_index.py), versioned with the image.
COPY api/bundle api/bundle

# 4. Code: the same encoder + fusion code and search page as the local app, plus the server.
COPY image_search/encoders.py image_search/fusion.py image_search/
COPY image_search/static/index.html image_search/static/index.html
COPY api/server.py api/web.py api/

EXPOSE 8080
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]
