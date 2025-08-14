###############################################################################
# Dockerfile  –  Imperceptible-ASR attack (CPU-only, x86-64) for Apple Silicon
# What it gives you:
#   • Ubuntu 18.04   • Python 2.7 + TensorFlow 1.14 (CPU wheel)
#   • Lingvo @ ICML-19 commit, pre-compiled custom ops
#
# NOTE: no pyroomacoustics / robust-attack dependencies.
###############################################################################

FROM --platform=linux/amd64 ubuntu:18.04

# ---------- system packages --------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        python2.7 python2.7-dev python-pip \
        git build-essential libsndfile1-dev g++ \
        sox wget ca-certificates openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/*

# ---------- Bazel (needed once to compile Lingvo ops) ------------------------
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-linux-x86_64 \
    -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# ---------- Python packages --------------------------------------------------
# ---------- python packages --------------------------------------------------
RUN pip install --upgrade pip==20.3.4 && \
    pip install setuptools==44.1.1 wheel==0.37.1

# Install core dependencies first
RUN pip install \
        numpy==1.16.6 \
        cython==0.29.36

# Install packages that depend on numpy
RUN pip install \
        scipy==1.2.3 \
        resampy==0.1.5 \
        scikit-learn==0.20.4 \
        joblib==0.14.1

# Install remaining packages
RUN pip install \
        tensorflow==1.14.0 \
        librosa==0.5.1 \
        audioread==2.1.8 \
        soundfile==0.10.3.post1 \
        absl-py==0.9.0 \
        matplotlib==2.2.5



# ---------- Lingvo (exact commit from ICML-19 paper) -------------------------
RUN git clone https://github.com/tensorflow/lingvo.git && \
    cd lingvo && git checkout 7d29b8f

# Compile Lingvo’s custom C/C++ ops once so first import is instant
RUN cd lingvo && bazel build //lingvo:trainer

# Expose Lingvo’s pure-Python modules
RUN pip install -e lingvo

ENV PYTHONPATH=/lingvo:$PYTHONPATH

WORKDIR /workspace
