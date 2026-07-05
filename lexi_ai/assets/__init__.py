"""Content-addressed derived-asset cache (Phase 4).

Assets (translation text, TTS clips) are keyed by a hash of the EXACT source
text plus a normalized param token — never by sense/word location. So themed vs
neutral text hash differently (distinct assets) and identical text dedups for
free: "each theme has its own translation/TTS" falls out of content addressing.
"""
