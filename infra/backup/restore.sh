#!/bin/sh
# Restore one explicitly selected Lexi dump. The product restore coordinator
# stops both services and restores Lexi before Pycil.
set -eu

: "${DATABASE_URL:?DATABASE_URL not set}"
: "${RESTIC_REPOSITORY:?RESTIC_REPOSITORY not set}"
: "${RESTIC_PASSWORD_FILE:?RESTIC_PASSWORD_FILE not set}"
: "${BACKUP_ID:?BACKUP_ID is required}"

restic dump "${SNAPSHOT:-latest}" "lexi-${BACKUP_ID}.dump" \
  | pg_restore --clean --if-exists --no-owner -d "$DATABASE_URL"
echo "[restore] lexi restore complete: ${BACKUP_ID}"
