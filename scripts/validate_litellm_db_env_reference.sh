#!/usr/bin/env bash
set -euo pipefail

# 在完全隔离的 Docker 网络中验证 v1.93+ DB model 重启后仍能解析
# api_key: os.environ/VARIABLE。只使用假密钥和本地 mock，不发起厂商请求。

LITELLM_IMAGE="${LITELLM_IMAGE:-litellm/litellm:v1.93.0@sha256:a1745e629abfb17d434426ff48b115f54f4f4c4a0f5af241de569e93c63c411e}"
POSTGRES_IMAGE="${POSTGRES_IMAGE:-postgres:15-alpine}"
PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.11-alpine}"
CURL_IMAGE="${CURL_IMAGE:-curlimages/curl:8.10.1}"
RUN_SUFFIX="fusion-litellm-env-${BASHPID}-${RANDOM}"
NETWORK="${RUN_SUFFIX}-net"
POSTGRES="${RUN_SUFFIX}-postgres"
MOCK="${RUN_SUFFIX}-mock"
PROXY="${RUN_SUFFIX}-proxy"
WORK_DIR="$(mktemp -d "/tmp/${RUN_SUFFIX}.XXXXXX")"
MASTER_KEY="sk-isolated-master"
SALT_KEY="sk-isolated-salt-key-must-remain-stable"
FAKE_PROVIDER_KEY="fake-provider-secret"

cleanup() {
  docker rm -f "$PROXY" "$MOCK" "$POSTGRES" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

cat >"$WORK_DIR/config.yaml" <<'YAML'
model_list: []
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: true
YAML

cat >"$WORK_DIR/mock.py" <<'PY'
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        if self.headers.get("authorization") != "Bearer fake-provider-secret":
            self.send_response(401)
            self.end_headers()
            return
        payload = {
            "id": "chatcmpl-isolated",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "env-reference-ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        return


HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
PY

cat >"$WORK_DIR/register.json" <<'JSON'
{
  "model_name": "env-reference-test",
  "litellm_params": {
    "model": "openai/fake-model",
    "api_base": "http://mock:8080/v1",
    "api_key": "os.environ/FAKE_PROVIDER_KEY"
  },
  "model_info": {
    "metadata": {
      "source": "fusion-isolated-validation"
    }
  }
}
JSON

cat >"$WORK_DIR/completion.json" <<'JSON'
{
  "model": "env-reference-test",
  "messages": [
    {"role": "user", "content": "ping"}
  ],
  "max_tokens": 8
}
JSON

docker network create "$NETWORK" >/dev/null
docker run -d --name "$POSTGRES" --network "$NETWORK" --network-alias postgres \
  -e POSTGRES_USER=litellm \
  -e POSTGRES_PASSWORD=litellm \
  -e POSTGRES_DB=litellm \
  "$POSTGRES_IMAGE" >/dev/null

for _ in $(seq 1 60); do
  if docker exec "$POSTGRES" pg_isready -U litellm -d litellm >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker exec "$POSTGRES" pg_isready -U litellm -d litellm >/dev/null

docker run -d --name "$MOCK" --network "$NETWORK" --network-alias mock \
  -v "$WORK_DIR/mock.py:/app/mock.py:ro" \
  "$PYTHON_IMAGE" python /app/mock.py >/dev/null

start_proxy() {
  docker run -d --name "$PROXY" --network "$NETWORK" --network-alias litellm \
    -v "$WORK_DIR/config.yaml:/app/config.yaml:ro" \
    -e DATABASE_URL="postgresql://litellm:litellm@postgres:5432/litellm" \
    -e LITELLM_MASTER_KEY="$MASTER_KEY" \
    -e LITELLM_SALT_KEY="$SALT_KEY" \
    -e STORE_MODEL_IN_DB=True \
    -e FAKE_PROVIDER_KEY="$FAKE_PROVIDER_KEY" \
    "$LITELLM_IMAGE" --config /app/config.yaml --port 4000 >/dev/null
  for _ in $(seq 1 120); do
    if docker run --rm --network "$NETWORK" "$CURL_IMAGE" \
      -fsS http://litellm:4000/health/liveliness >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  docker logs "$PROXY" >&2
  return 1
}

start_proxy
docker run --rm --network "$NETWORK" \
  -v "$WORK_DIR/register.json:/data/register.json:ro" \
  "$CURL_IMAGE" -fsS \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @/data/register.json \
  http://litellm:4000/model/new >/dev/null

docker rm -f "$PROXY" >/dev/null
start_proxy

RESULT="$(
  docker run --rm --network "$NETWORK" \
    -v "$WORK_DIR/completion.json:/data/completion.json:ro" \
    "$CURL_IMAGE" -fsS \
    -H "Authorization: Bearer $MASTER_KEY" \
    -H "Content-Type: application/json" \
    --data-binary @/data/completion.json \
    http://litellm:4000/chat/completions
)"

python3 -c '
import json
import sys

payload = json.loads(sys.stdin.read())
content = payload["choices"][0]["message"]["content"]
if content != "env-reference-ok":
    raise SystemExit("unexpected completion")
print(json.dumps({
    "status": "success",
    "image": "v1.93.0-fixed-digest",
    "db_reload_verified": True,
    "environment_reference_verified": True,
    "external_provider_requests": 0,
}, sort_keys=True))
' <<<"$RESULT"
