"""Minimal mTLS client for the canonical Lexi gRPC service.

Set LEXI_GRPC_TARGET, LEXI_GRPC_CA_FILE, LEXI_GRPC_CERT_FILE, and
LEXI_GRPC_KEY_FILE. The service identity is derived from this certificate.
"""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import grpc

from lexi_service.proto.lexi.v1 import lexi_pb2, lexi_pb2_grpc


def _read_env_file(name: str) -> bytes:
    path = os.environ.get(name)
    if not path:
        raise ValueError(f"{name} is required")
    return Path(path).read_bytes()


async def main(query: str) -> None:
    target = os.environ.get("LEXI_GRPC_TARGET")
    if not target:
        raise ValueError("LEXI_GRPC_TARGET is required (for example, lexi-grpc:9443)")
    credentials = grpc.ssl_channel_credentials(
        root_certificates=_read_env_file("LEXI_GRPC_CA_FILE"),
        private_key=_read_env_file("LEXI_GRPC_KEY_FILE"),
        certificate_chain=_read_env_file("LEXI_GRPC_CERT_FILE"),
    )
    async with grpc.aio.secure_channel(target, credentials) as channel:
        client = lexi_pb2_grpc.LexiServiceStub(channel)
        response = await client.Search(
            lexi_pb2.SearchRequest(query=query),
            metadata=(("x-request-id", uuid4().hex),),
        )
    for result in response.results:
        print(result.display, result.cambridge_id or result.lexi_word_id)


if __name__ == "__main__":
    import sys

    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "cat"))
