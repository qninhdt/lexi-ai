"""Shared mappings between application results/errors and transport models."""

from google.protobuf.json_format import MessageToDict

from lexi_ai.read_models import Entry, SearchResult
from lexi_service.application.errors import ErrorCode, PublicError
from lexi_service.proto.lexi.v1 import lexi_pb2

HTTP_STATUS = {
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.VALIDATION: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.DEADLINE_EXCEEDED: 408,
    ErrorCode.PRECONDITION_FAILED: 412,
    ErrorCode.INTERNAL: 500,
}


def error_body(error: PublicError) -> dict:
    return {
        "error": {
            "code": error.code.value,
            "message": error.message,
            "retryable": error.retryable,
            "incident_id": error.incident_id,
        }
    }


def search_target(value: SearchResult) -> lexi_pb2.SearchTarget:
    return lexi_pb2.SearchTarget(
        display=value.display,
        entry_type=value.entry_type or "",
        score=value.score,
        lexi_word_id=value.lexi_word_id or 0,
        cambridge_id=value.cambridge_id or 0,
        gloss=value.gloss or "",
    )


def entry(value: Entry) -> lexi_pb2.Entry:
    return lexi_pb2.Entry(
        word_id=value.word_id,
        display=value.display,
        norm=value.norm,
        status=value.status,
        entry_type=value.entry_type or "",
        senses=[
            lexi_pb2.Sense(
                definition=s.definition,
                tier=s.tier,
                pos=s.pos or "",
                cefr_level=s.cefr_level or "",
                examples=s.examples,
            )
            for s in value.senses
        ],
    )


def json_message(value) -> dict:
    return MessageToDict(value, preserving_proto_field_name=True)
