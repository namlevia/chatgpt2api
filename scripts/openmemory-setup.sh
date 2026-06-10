#!/usr/bin/env bash
# Setup OpenMemory trên server — lớp ký ức dài hạn cho chatgpt2api
# ("đổi account không mất dòng suy nghĩ").
#
# Image build sẵn bởi CI (.github/workflows/openmemory-build.yml) →
# ghcr.io/tritue2011/openmemory:latest. Server CHỈ PULL, không build.
#
# Chạy 1 lần trên .38:
#   OM_API_KEY=<key mạnh tự đặt> [GEMINI_API_KEY=<key AI Studio>] \
#     bash scripts/openmemory-setup.sh
#
# Sau đó bật trong chatgpt2api config (UI Settings → providers):
#   providers.openmemory = {
#     "enabled": true,
#     "base_url": "http://172.16.10.38:8081",
#     "api_key": "<OM_API_KEY ở trên>"
#   }
# KHÔNG commit key vào repo — key chỉ nằm trong /opt/openmemory/.env.

set -euo pipefail

OM_DIR=/opt/openmemory
HOST_PORT=${HOST_PORT:-8081}
IMAGE=${IMAGE:-ghcr.io/tritue2011/openmemory:latest}

if [ -z "${OM_API_KEY:-}" ]; then
  echo "ERROR: cần OM_API_KEY=... (key tự đặt cho OpenMemory)" >&2
  exit 1
fi

mkdir -p "$OM_DIR/data"

# .env — chứa secret, chỉ nằm trên server
cat > "$OM_DIR/.env" <<EOF
OM_API_KEY=$OM_API_KEY
OM_PORT=8080
OM_METADATA_BACKEND=sqlite
OM_DB_PATH=/data/openmemory.sqlite
OM_VECTOR_BACKEND=sqlite
OM_EMBEDDINGS=synthetic
OM_EMBEDDING_FALLBACK=synthetic
OM_TIER=hybrid
OM_MIN_SCORE=0.3
EOF
# Có key Gemini → dùng embeddings gemini (recall tốt hơn synthetic)
if [ -n "${GEMINI_API_KEY:-}" ]; then
  sed -i 's/^OM_EMBEDDINGS=.*/OM_EMBEDDINGS=gemini/' "$OM_DIR/.env"
  echo "GEMINI_API_KEY=$GEMINI_API_KEY" >> "$OM_DIR/.env"
fi
chmod 600 "$OM_DIR/.env"

docker pull "$IMAGE"
docker rm -f openmemory 2>/dev/null || true
docker run -d \
  --name openmemory \
  --restart unless-stopped \
  --env-file "$OM_DIR/.env" \
  -p "$HOST_PORT:8080" \
  -v "$OM_DIR/data:/data" \
  "$IMAGE"

echo "[openmemory] waiting for health..."
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$HOST_PORT/health" || true)
  if [ "$code" = "200" ]; then
    echo "[openmemory] SUCCESS — http://127.0.0.1:$HOST_PORT/health OK"
    exit 0
  fi
  sleep 2
done
echo "[openmemory] FAILED — health không lên sau 60s, xem: docker logs openmemory" >&2
exit 1
