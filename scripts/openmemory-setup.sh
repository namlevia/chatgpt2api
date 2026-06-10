#!/usr/bin/env bash
# Setup OpenMemory (CaviraOSS/OpenMemory) trên server — lớp ký ức dài hạn
# cho chatgpt2api ("đổi account không mất dòng suy nghĩ").
#
# Không có image dựng sẵn → build từ source. Chạy 1 lần trên .38:
#   OM_API_KEY=<key mạnh tự đặt> [GEMINI_API_KEY=<key AI Studio>] \
#     bash deploy/openmemory-setup.sh
#
# Sau đó bật trong chatgpt2api config (UI Settings → providers):
#   providers.openmemory = {
#     "enabled": true,
#     "base_url": "http://172.16.10.38:8081",
#     "api_key": "<OM_API_KEY ở trên>"
#   }
# KHÔNG commit key vào repo — key chỉ nằm trong /opt/openmemory/src/.env.

set -euo pipefail

OM_DIR=/opt/openmemory
SRC_DIR=$OM_DIR/src
HOST_PORT=${HOST_PORT:-8081}

if [ -z "${OM_API_KEY:-}" ]; then
  echo "ERROR: cần OM_API_KEY=... (key tự đặt cho OpenMemory)" >&2
  exit 1
fi

mkdir -p "$OM_DIR"
if [ -d "$SRC_DIR/.git" ]; then
  git -C "$SRC_DIR" pull --ff-only
else
  git clone --depth 1 https://github.com/CaviraOSS/OpenMemory.git "$SRC_DIR"
fi

# .env — chứa secret, chỉ nằm trên server
cat > "$SRC_DIR/.env" <<EOF
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
  sed -i 's/^OM_EMBEDDINGS=.*/OM_EMBEDDINGS=gemini/' "$SRC_DIR/.env"
  echo "GEMINI_API_KEY=$GEMINI_API_KEY" >> "$SRC_DIR/.env"
fi
chmod 600 "$SRC_DIR/.env"

# Compose riêng (không dùng compose gốc của repo): tự kiểm soát port host
# (8080 có thể bận) + persist data ra /opt/openmemory/data
mkdir -p "$OM_DIR/data"
cat > "$SRC_DIR/docker-compose.chatgpt2api.yml" <<EOF
services:
  openmemory:
    build:
      context: ./packages/openmemory-js
      dockerfile: Dockerfile
    container_name: openmemory
    restart: unless-stopped
    env_file: .env
    ports:
      - "$HOST_PORT:8080"
    volumes:
      - $OM_DIR/data:/data
EOF

cd "$SRC_DIR"
docker compose -f docker-compose.chatgpt2api.yml up --build -d

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
