"""Everything that leaves this process, in one package.

An OpenAI-compatible chat endpoint, an OpenAI-compatible speech endpoint, a
Cambridge SQLite file, and WordNet. Each module binds one of them to the shape the
library needs, and each is injectable so a test supplies a fake not a credential.
"""
