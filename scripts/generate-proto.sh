#!/usr/bin/env sh
set -eu

uv run python -m grpc_tools.protoc \
  -Ilexi_service/proto=proto \
  --python_out=. \
  --pyi_out=. \
  --grpc_python_out=. \
  lexi_service/proto/lexi/v1/lexi.proto
