FROM prism-llmservingsim-m8:2c2042ce

COPY . /app/prism
WORKDIR /app/prism/third_party/LLMServingSim

# Build the exact recursively pinned ASTRA-Sim tree with upstream's script.
RUN bash scripts/compile.sh

# Chakra's generated protobuf sources at this pin require runtime 7.35.1.
RUN pip3 install --no-cache-dir protobuf==7.35.1
