#!/bin/sh
# Lexi database backup. Product-level coordination is owned by Pycil's
# scripts/backup-product.sh so both dumps share one BACKUP_ID.
set -eu

: "${DATABASE_URL:?DATABASE_URL not set}"
: "${RESTIC_REPOSITORY:?RESTIC_REPOSITORY not set}"
: "${RESTIC_PASSWORD_FILE:?RESTIC_PASSWORD_FILE not set}"

backup_id="${BACKUP_ID:-lexi-$(date -u +%Y%m%dT%H%M%SZ)}"
restic snapshots --tag lexi-pg- >/dev/null 2>&1 || restic init
pg_dump -Fc -d "$DATABASE_URL" | restic backup \
  --tag "lexi-pg-${backup_id}" --stdin --stdin-filename "lexi-${backup_id}.dump"
echo "[backup] lexi backup complete: ${backup_id}"
