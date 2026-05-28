ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
FROM ${BASE_IMAGE}

# OS deps
# NOTE: gstreamer*, python3-gi, gir1.2-*, libjansson4, libjson-glib, libyaml-cpp,
# libgstrtspserver are already in the DeepStream 9.0 base image — do not re-add.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git wget build-essential cmake \
    python3.12-venv python3-dev \
    libboost-python-dev \
    rapidjson-dev \
    libcudss0-cuda-13 \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# DeepStream extras
RUN bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh

# Create venv with system-site-packages (for gi, DeepStream bindings)
RUN python3 -m venv --system-site-packages /opt/pyds_env && \
    /opt/pyds_env/bin/pip install --upgrade pip

# Build & install DeepStream Python bindings (no pre-built wheel for DS 9.0)
RUN bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --build-bindings && \
    /opt/pyds_env/bin/pip install \
        /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings/dist/pyds-*.whl

ENV VIRTUAL_ENV=/opt/pyds_env
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# PyTorch with CUDA 12.8
RUN /opt/pyds_env/bin/pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Project requirements
COPY docker/requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt

# Entrypoint
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /deepstream
CMD [ "/opt/entrypoint.sh", "-r" ]
