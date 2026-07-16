"""Verified service identity shared by commands and service ports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant: str | None = None

    @property
    def ownership_key(self) -> tuple[str, str | None]:
        return self.subject, self.tenant
