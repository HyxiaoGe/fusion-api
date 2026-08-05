#!/usr/bin/env bash
set -euo pipefail

image_name="${1:?缺少 API 镜像名}"
adapter_image_name="${2:?缺少 FlyAI Adapter 镜像名}"
image_tag="${3:?缺少镜像标签}"
image="${image_name}:${image_tag}"
adapter_image="${adapter_image_name}:${image_tag}"

docker build --target production -t "${image}" .

docker run --rm "${image}" sh -lc "timeout 300s python -m pip install --default-timeout=30 --no-cache-dir -r requirements-ci.txt && python scripts/check_architecture.py && ruff check . && timeout 270s python -u -m unittest discover -s test -t . -v"

docker build --target test -t "${adapter_image}-test" ./flyai-adapter
docker build --target production -t "${adapter_image}" ./flyai-adapter
