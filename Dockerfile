# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12.10-slim-bookworm

FROM ${PYTHON_IMAGE} AS build-test

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /source
COPY . .

# Exercise the standalone C++ contract independently from the Python wheel.
RUN cmake -S cpp -B /build/native-debug -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DPRISM_BUILD_TESTS=ON \
        -DPRISM_WARNINGS_AS_ERRORS=ON \
    && cmake --build /build/native-debug --parallel \
    && ctest --test-dir /build/native-debug --output-on-failure

# Prism already has straightforward, separate Linux sanitizer configurations.
RUN cmake -S cpp -B /build/native-asan -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DPRISM_BUILD_TESTS=ON \
        -DPRISM_WARNINGS_AS_ERRORS=ON \
        -DPRISM_ENABLE_ASAN=ON \
    && cmake --build /build/native-asan --parallel \
    && ctest --test-dir /build/native-asan --output-on-failure

RUN cmake -S cpp -B /build/native-ubsan -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DPRISM_BUILD_TESTS=ON \
        -DPRISM_WARNINGS_AS_ERRORS=ON \
        -DPRISM_ENABLE_UBSAN=ON \
    && cmake --build /build/native-ubsan --parallel \
    && ctest --test-dir /build/native-ubsan --output-on-failure

# Build the existing scikit-build-core wheel. Its private pybind11 extension
# links the same prism_storage CMake target tested above.
RUN python -m pip wheel \
        --constraint container/constraints.txt \
        --wheel-dir /app-wheels \
        . \
    && python -m pip wheel \
        --constraint container/constraints.txt \
        --wheel-dir /test-wheels \
        pytest==9.1.1 \
    && python -m pip install \
        --no-index \
        --find-links /app-wheels \
        prism-storage==1.0.0 \
    && python -m pip install \
        --no-index \
        --find-links /test-wheels \
        pytest==9.1.1

RUN python -c "import prism, prism._native, prism.native; print(prism.__version__)" \
    && python -m prism.native.cli --fixture --output-dir /tmp/prism-native-smoke \
    && python -m pytest -q -o pythonpath= \
    && python -m pip check


FROM ${PYTHON_IMAGE} AS runtime

ARG PRISM_GIT_REVISION=unknown
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PRISM_GIT_REVISION=${PRISM_GIT_REVISION} \
    PRISM_INPUT_ROOT=/input \
    PRISM_OUTPUT_ROOT=/output \
    PRISM_NATIVE_BUILD_TYPE=Release \
    PRISM_NATIVE_COMPILER_ID=GNU \
    PRISM_NATIVE_COMPILER_VERSION=12.2.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 prism \
    && useradd --system --uid 10001 --gid prism --no-create-home prism \
    && mkdir /input /output \
    && chown prism:prism /output

COPY --from=build-test /app-wheels /wheels
RUN python -m pip install \
        --no-index \
        --find-links /wheels \
        prism-storage==1.0.0 \
    && rm -rf /wheels

USER 10001:10001
WORKDIR /output

CMD ["prism-container-run", "--help"]
