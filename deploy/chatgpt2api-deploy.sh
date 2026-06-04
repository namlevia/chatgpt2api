#!/bin/bash
set -e

IMAGE="ghcr.io/tritue2011/chatgpt2api:latest"
CONTAINER="chatgpt2api"
DATA_DIR="/opt/chatgpt2api-data"
PORT="3030"
AUTH_KEY="$(docker inspect $CONTAINER --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep CHATGPT2API_AUTH_KEY | cut -d= -f2-)"

echo "[deploy] Cleaning up old images..."
docker image prune -af 2>/dev/null || true
docker container prune -f 2>/dev/null || true

df -h / | tail -1

echo "[deploy] Saving current image SHA as fallback..."
PREV_SHA=$(docker inspect $CONTAINER --format '{{.Image}}' 2>/dev/null || echo "")
echo "Previous SHA: $PREV_SHA"

echo "[deploy] Pulling new image..."
docker pull $IMAGE

echo "[deploy] Stopping old container..."
docker stop $CONTAINER 2>/dev/null || true
docker rm -f $CONTAINER 2>/dev/null || true

echo "[deploy] Starting new container..."
docker run -d \
  --name $CONTAINER \
  --restart unless-stopped \
  -p ${PORT}:80 \
  -v ${DATA_DIR}:/app/data \
  -e CHATGPT2API_AUTH_KEY="$AUTH_KEY" \
  -e STORAGE_BACKEND=json \
  $IMAGE

echo "[deploy] Waiting for healthcheck..."
for i in $(seq 1 12); do
  sleep 5
  HTTP=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/version" || echo "000")
  echo "  attempt $i: HTTP $HTTP"
  if [ "$HTTP" = "200" ]; then
    echo "[deploy] SUCCESS — container healthy"
    exit 0
  fi
done

echo "[deploy] HEALTHCHECK FAILED — rolling back to previous image..."
docker stop $CONTAINER 2>/dev/null || true
docker rm -f $CONTAINER 2>/dev/null || true

if [ -n "$PREV_SHA" ]; then
  docker run -d \
    --name $CONTAINER \
    --restart unless-stopped \
    -p ${PORT}:80 \
    -v ${DATA_DIR}:/app/data \
    -e CHATGPT2API_AUTH_KEY="$AUTH_KEY" \
    -e STORAGE_BACKEND=json \
    "$PREV_SHA"
  echo "[deploy] Rolled back to $PREV_SHA"
else
  echo "[deploy] No previous image to roll back to!"
fi

exit 1
