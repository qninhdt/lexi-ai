"""Content fingerprints that give domain records a stable identity."""

import hashlib


def sense_content_hash(definition: str) -> str:
    """Stable content fingerprint of a target sense.

    Stamped on ``sense_relation.target_hash`` at resolve time and re-checked on
    read: if the target sense's definition later changes (regenerate), the stored
    hash no longer matches and the edge is treated as unresolved rather than
    silently pointing at a mutated meaning. The definition is the load-bearing
    meaning carrier, so it alone keys the hash.
    """
    return hashlib.sha256(definition.encode("utf-8")).hexdigest()
