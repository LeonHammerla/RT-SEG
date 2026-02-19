# syntax=docker/dockerfile:1.5

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.12.3

# =========================
# 1) Build CPython 3.12.3
# =========================
FROM ${CUDA_IMAGE} AS pybuild
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev tk-dev uuid-dev \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o /tmp/Python.tgz \
  && tar -xzf /tmp/Python.tgz -C /tmp \
  && cd "/tmp/Python-${PYTHON_VERSION}" \
  && ./configure --prefix=/opt/python --with-ensurepip=install --enable-shared \
  && make -j"$(nproc)" \
  && make install \
  # 👇 make libpython discoverable
  && echo "/opt/python/lib" > /etc/ld.so.conf.d/python312.conf \
  && ldconfig \
  && /opt/python/bin/python3 -m pip install --upgrade pip setuptools wheel \
  && rm -rf /tmp/*

RUN /opt/python/bin/python3 -c "import sys; assert sys.version.startswith('${PYTHON_VERSION}'), sys.version"


# =========================
# 2) Build wheels (native deps like hdbscan)
# =========================
FROM ${CUDA_IMAGE} AS wheelbuild
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app/RT-SEG

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=pybuild /opt/python /opt/python
RUN echo "/opt/python/lib" > /etc/ld.so.conf.d/python312.conf && ldconfig
ENV PATH="/opt/python/bin:${PATH}"

COPY requirements.txt /app/RT-SEG/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip wheel --wheel-dir /wheels -r /app/RT-SEG/requirements.txt


# =========================
# 3) Runtime (small)
# =========================
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/opt/cache \
    HF_HOME=/opt/cache/huggingface \
    TRANSFORMERS_CACHE=/opt/cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/opt/cache/sentence-transformers \
    NLTK_DATA=/opt/cache/nltk \
    SPACY_HOME=/opt/cache/spacy

WORKDIR /app/RT-SEG

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 zlib1g libbz2-1.0 libreadline8 libsqlite3-0 libffi8 liblzma5 tk uuid-runtime \
  && rm -rf /var/lib/apt/lists/*

COPY --from=pybuild /opt/python /opt/python
RUN echo "/opt/python/lib" > /etc/ld.so.conf.d/python312.conf && ldconfig
ENV PATH="/opt/python/bin:${PATH}"

COPY --from=wheelbuild /wheels /wheels
COPY requirements.txt /app/RT-SEG/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-index --find-links=/wheels -r /app/RT-SEG/requirements.txt \
    && rm -rf /wheels

COPY src /app/RT-SEG/src
COPY data/tui_css /app/RT-SEG/data/tui_css
COPY data/prompts.json /app/RT-SEG/data/prompts.json
COPY data/example_traces.json /app/RT-SEG/data/example_traces.json

# if your app loads CSS from /data/tui_css/app.css
RUN ln -s /app/RT-SEG/data /data

RUN python3 -c "import pathlib; p=pathlib.Path('src/preload_models.py'); print('preload script exists:', p.exists())" \
    && (python3 src/preload_models.py || true)

CMD ["python3", "src/tui.py"]
