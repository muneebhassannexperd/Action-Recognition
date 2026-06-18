FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY utils/ utils/
COPY data/ data/
COPY service/ service/
COPY scripts/download_models.sh scripts/download_models.sh

ARG DOWNLOAD_WEIGHTS=true
RUN if [ "$DOWNLOAD_WEIGHTS" = "true" ]; then \
        mkdir -p pretrained_model/posec3d && \
        curl -L --fail -o pretrained_model/posec3d/joint.pth \
            "http://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth"; \
    fi

ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
# ORGANIZATION_ID and CAMERA_ID are required at runtime (no defaults).
# DEVICE_ID is optional (set for edge deployment).
ENV DEVICE=cuda
ENV ACTION_MODEL=posec3d
ENV LOG_LEVEL=INFO

CMD ["python", "service/posec3d_service.py"]
