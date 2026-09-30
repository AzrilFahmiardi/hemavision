FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY app/ app/
COPY src/ src/
COPY configs/ configs/
COPY artifacts/conjunctiva/segmentation_unet.pt artifacts/conjunctiva/segmentation_unet.pt
COPY artifacts/conjunctiva/embedding_backbone_dinov2_small.pt artifacts/conjunctiva/embedding_backbone_dinov2_small.pt
COPY artifacts/conjunctiva/multitask_full_fusion_dinov2_small_fold0.pt artifacts/conjunctiva/multitask_full_fusion_dinov2_small_fold0.pt
COPY artifacts/conjunctiva/fusion_stats.json artifacts/conjunctiva/fusion_stats.json
COPY artifacts/palm/embedding_backbone_siglip_base_patch16_224.pt artifacts/palm/embedding_backbone_siglip_base_patch16_224.pt
COPY artifacts/palm/multitask_path_b_deep_siglip_base_patch16_224_fold0.pt artifacts/palm/multitask_path_b_deep_siglip_base_patch16_224_fold0.pt
COPY artifacts/palm/fusion_stats.json artifacts/palm/fusion_stats.json
COPY artifacts/palm/hand_landmarker.task artifacts/palm/hand_landmarker.task
COPY artifacts/nail/embedding_backbone_siglip_base_patch16_224.pt artifacts/nail/embedding_backbone_siglip_base_patch16_224.pt
COPY artifacts/nail/multitask_path_b_deep_siglip_base_patch16_224_fold0.pt artifacts/nail/multitask_path_b_deep_siglip_base_patch16_224_fold0.pt
COPY artifacts/nail/fusion_stats.json artifacts/nail/fusion_stats.json

ENV PORT=8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
