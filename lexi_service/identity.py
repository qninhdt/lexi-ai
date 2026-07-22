"""Verified service identity shared by commands and service ports."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant: str | None = None

    @property
    def ownership_key(self) -> tuple[str, str | None]:
        return self.subject, self.tenant

    @property
    def provider_scope(self) -> str:
        """Stable, collision-free text key for provider concurrency gates."""
        return json.dumps([self.subject, self.tenant], separators=(",", ":"))
