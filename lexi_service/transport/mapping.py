"""Shared mappings between application results/errors and transport models."""

from google.protobuf.json_format import MessageToDict

from lexi_ai.read_models import Entry, Question, SearchResult, SenseView
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
            sense(s)
            for s in value.senses
        ],
    )


def sense(value: SenseView) -> lexi_pb2.Sense:
    return lexi_pb2.Sense(
        definition=value.definition,
        tier=value.tier,
        pos=value.pos or "",
        cefr_level=value.cefr_level or "",
        examples=value.examples,
        sense_id=value.sense_id or 0,
    )


def json_message(value) -> dict:
    return MessageToDict(value, preserving_proto_field_name=True)


def question_metadata(value: Question) -> dict:
    """Stable metadata for a service-owned, persisted learner question."""
    if value.id is None or value.sense_id is None:
        raise ValueError("learner questions require persisted id and sense id")
    return {
        "question_id": value.id,
        "sense_id": value.sense_id,
        "format": value.format,
        "answer_kind": value.answer_kind,
    }


def question_presentation(value: Question) -> dict:
    """Return the allowlisted learner projection, never the grading payload.

    This is intentionally format-specific rather than a denylist over arbitrary
    plugin JSON: new answer-bearing fields cannot leak merely because a plugin
    introduced them.  Answer material stays in Lexi and is used only by grade.
    """
    result = question_metadata(value)
    payload = value.payload
    if value.format in {"definition_mcq", "contextual_mcq", "pronunciation_mcq"}:
        public = {"stem": payload["stem"], "options": payload["options"]}
    elif value.format in {"cloze", "collocation_fill"}:
        public = {"stem_with_blank": payload["stem_with_blank"]}
    elif value.format == "use_in_sentence":
        public = {"prompt": payload["prompt"]}
    elif value.format == "listening":
        public = {
            "prompt": payload["prompt"],
            "audio_ref": payload["audio_ref"],
            "options": payload["options"],
        }
    elif value.format == "spelling":
        public = {"prompt": payload["prompt"], "audio_ref": payload["audio_ref"]}
    elif value.format == "matching":
        public = {
            "prompt": payload["prompt"],
            "lefts": payload["lefts"],
            "rights": payload["rights"],
        }
    else:
        raise ValueError("unsupported question format")
    result["presentation"] = {"type": value.format, **public}
    return result
