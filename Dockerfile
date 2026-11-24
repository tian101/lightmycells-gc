# Container image for grand-challenge.org, built by the platform from this repo.
#
# Grand Challenge links this repository, builds the Dockerfile below on a tag
# push, and runs the resulting image. The four checkpoints cannot live in git
# (4 x 208 MB, over GitHub's 100 MB per-file limit), so they are downloaded from
# a release asset at build time and validated before the layer is committed.
#
# If the release tag or an asset name changes, bump --tag here.
ARG BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# tifffile writes the OME-TIFF outputs; imagecodecs is needed to *decode*
# compressed inputs (LZW / Deflate / JPEG). SimpleITK only adds the .mha variant
# of the same interface, so a failure there degrades instead of breaking the build.
RUN pip install --no-cache-dir \
        "tifffile>=2024.5.10" \
        "imagecodecs>=2024.1.1" \
        "pillow>=10.0" \
 && (pip install --no-cache-dir "SimpleITK>=2.3.1" \
     || echo "SimpleITK unavailable: .mha inputs will be rejected, .tif works")

COPY fetch_weights.py /opt/fetch_weights.py
RUN python /opt/fetch_weights.py --dest /opt/weights --tag v1

WORKDIR /opt/app
COPY app/ /opt/app/

# grand-challenge.org refuses to import an image that runs as root, so the
# entrypoint runs unprivileged. Code and weights stay root-owned and
# world-readable; only the home directory and /output need to be writable.
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser \
 && mkdir -p /input /output \
 && chown -R appuser:appuser /home/appuser /output

USER appuser
ENV HOME=/home/appuser

ENTRYPOINT ["python", "/opt/app/gc_entrypoint.py"]