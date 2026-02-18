ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:8.0-triton-multiarch
FROM ${BASE_IMAGE}

# OS deps
RUN apt-get update && apt-get install -y \
    curl git zip wget build-essential autoconf autogen unzip \
    libboost-all-dev rapidjson-dev graphviz ffmpeg libsm6 libxext6 cmake \
    python3.12-venv \
    libcudss0-cuda-12 \
    libssl-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstrtspserver-1.0-0 \
    libjansson4 \
    libjson-glib-1.0-0 \
    libyaml-cpp-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    python3-dev \
    libx11-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-add-repository ppa:mosquitto-dev/mosquitto-ppa && \
    apt-get update && \
    apt-get install -y libmosquitto-dev

# DeepStream extras (NVIDIA script)
RUN bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh

# -------- Create our venv with system-site-packages (for gi, DeepStream bindings) --------
RUN python3 -m venv --system-site-packages /opt/pyds_env && \
    /opt/pyds_env/bin/pip install --upgrade pip

# Install DeepStream Python bindings (NVIDIA script creates its own venv)
# Keep calls explicit to avoid polluting our venv.
RUN bash /opt/nvidia/deepstream/deepstream/user_deepstream_python_apps_install.sh --version 1.2.2

# Install pyds wheel into OUR venv explicitly
RUN /opt/pyds_env/bin/pip install \
    /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/pyds-1.2.2-cp312-cp312-linux_x86_64.whl

# Install PyTorch for x86 with CUDA 12.8
RUN /opt/pyds_env/bin/pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

ENV VIRTUAL_ENV=/opt/pyds_env
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Project requirements (these will go into /opt/pyds_env now)
COPY docker/requirements.txt /opt/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/requirements.txt

# Upgrade CMake to 3.31.8 (required for Triton Python backend build)
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.31.8-linux-x86_64.tar.gz && \
    rm -rf /usr/local/man && \
    cp -rf cmake-3.31.8-linux-x86_64/* /usr/local/ && \
    rm -rf cmake-3.31.8-linux-x86_64* && \
    cmake --version

# Scripts & Triton Python backend setup (now uses venv python/pip by default)
COPY scripts/ /opt/scripts
RUN chmod +x /opt/scripts/*.sh && /opt/scripts/install_triton_python_backend_gpu.sh

# Entrypoint
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /deepstream
CMD [ "/opt/entrypoint.sh", "-r" ]