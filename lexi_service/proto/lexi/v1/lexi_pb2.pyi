from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResponseMeta(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    query: str
    def __init__(self, query: _Optional[str] = ...) -> None: ...

class SearchTarget(_message.Message):
    __slots__ = ("display", "entry_type", "score", "lexi_word_id", "cambridge_id", "gloss")
    DISPLAY_FIELD_NUMBER: _ClassVar[int]
    ENTRY_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    LEXI_WORD_ID_FIELD_NUMBER: _ClassVar[int]
    CAMBRIDGE_ID_FIELD_NUMBER: _ClassVar[int]
    GLOSS_FIELD_NUMBER: _ClassVar[int]
    display: str
    entry_type: str
    score: float
    lexi_word_id: int
    cambridge_id: int
    gloss: str
    def __init__(self, display: _Optional[str] = ..., entry_type: _Optional[str] = ..., score: _Optional[float] = ..., lexi_word_id: _Optional[int] = ..., cambridge_id: _Optional[int] = ..., gloss: _Optional[str] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("meta", "results")
    META_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    meta: ResponseMeta
    results: _containers.RepeatedCompositeFieldContainer[SearchTarget]
    def __init__(self, meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., results: _Optional[_Iterable[_Union[SearchTarget, _Mapping]]] = ...) -> None: ...

class LookupRequest(_message.Message):
    __slots__ = ("word_id",)
    WORD_ID_FIELD_NUMBER: _ClassVar[int]
    word_id: int
    def __init__(self, word_id: _Optional[int] = ...) -> None: ...

class Sense(_message.Message):
    __slots__ = ("definition", "tier", "pos", "cefr_level", "examples", "sense_id")
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    POS_FIELD_NUMBER: _ClassVar[int]
    CEFR_LEVEL_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    SENSE_ID_FIELD_NUMBER: _ClassVar[int]
    definition: str
    tier: str
    pos: str
    cefr_level: str
    examples: _containers.RepeatedScalarFieldContainer[str]
    sense_id: int
    def __init__(self, definition: _Optional[str] = ..., tier: _Optional[str] = ..., pos: _Optional[str] = ..., cefr_level: _Optional[str] = ..., examples: _Optional[_Iterable[str]] = ..., sense_id: _Optional[int] = ...) -> None: ...

class Entry(_message.Message):
    __slots__ = ("word_id", "display", "norm", "status", "entry_type", "senses")
    WORD_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_FIELD_NUMBER: _ClassVar[int]
    NORM_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ENTRY_TYPE_FIELD_NUMBER: _ClassVar[int]
    SENSES_FIELD_NUMBER: _ClassVar[int]
    word_id: int
    display: str
    norm: str
    status: str
    entry_type: str
    senses: _containers.RepeatedCompositeFieldContainer[Sense]
    def __init__(self, word_id: _Optional[int] = ..., display: _Optional[str] = ..., norm: _Optional[str] = ..., status: _Optional[str] = ..., entry_type: _Optional[str] = ..., senses: _Optional[_Iterable[_Union[Sense, _Mapping]]] = ...) -> None: ...

class EntryResponse(_message.Message):
    __slots__ = ("meta", "entry")
    META_FIELD_NUMBER: _ClassVar[int]
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    meta: ResponseMeta
    entry: Entry
    def __init__(self, meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., entry: _Optional[_Union[Entry, _Mapping]] = ...) -> None: ...

class GetSensesRequest(_message.Message):
    __slots__ = ("sense_ids",)
    SENSE_IDS_FIELD_NUMBER: _ClassVar[int]
    sense_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, sense_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class SensesResponse(_message.Message):
    __slots__ = ("meta", "senses")
    META_FIELD_NUMBER: _ClassVar[int]
    SENSES_FIELD_NUMBER: _ClassVar[int]
    meta: ResponseMeta
    senses: _containers.RepeatedCompositeFieldContainer[Sense]
    def __init__(self, meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., senses: _Optional[_Iterable[_Union[Sense, _Mapping]]] = ...) -> None: ...

class SubmitGenerateRequest(_message.Message):
    __slots__ = ("target", "reference_dataset_fingerprint", "payload_version")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_DATASET_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_VERSION_FIELD_NUMBER: _ClassVar[int]
    target: SearchTarget
    reference_dataset_fingerprint: str
    payload_version: int
    def __init__(self, target: _Optional[_Union[SearchTarget, _Mapping]] = ..., reference_dataset_fingerprint: _Optional[str] = ..., payload_version: _Optional[int] = ...) -> None: ...

class SubmitTranslationRequest(_message.Message):
    __slots__ = ("source_kind", "source_id", "language", "source_hash", "reference_dataset_fingerprint", "payload_version")
    SOURCE_KIND_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_HASH_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_DATASET_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_VERSION_FIELD_NUMBER: _ClassVar[int]
    source_kind: str
    source_id: int
    language: str
    source_hash: str
    reference_dataset_fingerprint: str
    payload_version: int
    def __init__(self, source_kind: _Optional[str] = ..., source_id: _Optional[int] = ..., language: _Optional[str] = ..., source_hash: _Optional[str] = ..., reference_dataset_fingerprint: _Optional[str] = ..., payload_version: _Optional[int] = ...) -> None: ...

class GetJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class Job(_message.Message):
    __slots__ = ("job_id", "status", "deduplicated", "operation", "result_json", "error_code")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DEDUPLICATED_FIELD_NUMBER: _ClassVar[int]
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: str
    deduplicated: bool
    operation: str
    result_json: str
    error_code: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[str] = ..., deduplicated: _Optional[bool] = ..., operation: _Optional[str] = ..., result_json: _Optional[str] = ..., error_code: _Optional[str] = ...) -> None: ...

class JobResponse(_message.Message):
    __slots__ = ("meta", "job")
    META_FIELD_NUMBER: _ClassVar[int]
    JOB_FIELD_NUMBER: _ClassVar[int]
    meta: ResponseMeta
    job: Job
    def __init__(self, meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., job: _Optional[_Union[Job, _Mapping]] = ...) -> None: ...

class StatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatsResponse(_message.Message):
    __slots__ = ("meta", "words_by_status", "senses", "examples", "tags")
    class WordsByStatusEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    META_FIELD_NUMBER: _ClassVar[int]
    WORDS_BY_STATUS_FIELD_NUMBER: _ClassVar[int]
    SENSES_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    meta: ResponseMeta
    words_by_status: _containers.ScalarMap[str, int]
    senses: int
    examples: int
    tags: int
    def __init__(self, meta: _Optional[_Union[ResponseMeta, _Mapping]] = ..., words_by_status: _Optional[_Mapping[str, int]] = ..., senses: _Optional[int] = ..., examples: _Optional[int] = ..., tags: _Optional[int] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...
