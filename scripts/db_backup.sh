#!/usr/bin/env bash
#
# Dump the Dockerized PostgreSQL database to a file on this host (R1-T5).
#
# Usage:
#   scripts/db_backup.sh                     # backups/deepguard-<utc timestamp>.dump
#   scripts/db_backup.sh path/to/file.dump   # an explicit destination
#   DATABASE=deepguard_scratch scripts/db_backup.sh   # a database other than the default
#
# The dump is `pg_dump -Fc`: PostgreSQL's own compressed custom format, which is what
# `pg_restore` reads. Plain SQL would be human-readable and would also have to be restored by
# piping it into `psql`, with no selective restore, no parallelism and no way to check what is
# in the file without reading all of it. `pg_restore --list` answers that question about a
# custom-format dump in a second.
#
# Nothing is installed on the host: `pg_dump` runs inside the `postgres` container, so the
# client and the server are the same version by construction. A host client one major version
# behind the server refuses to dump at all, and that mismatch is the usual reason a backup
# script stops working after an image bump.
set -euo pipefail

# The repository root, whatever directory this was called from — `docker compose` has to run
# where the compose file and `.env` are.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "db_backup: .env not found in $ROOT. Run 'cp .env.example .env' first." >&2
  exit 1
fi

# The credentials, read from the one file the stack is already configured by. Sourced in a
# subshell-safe way: `set -a` exports what the file assigns so the values are available below
# without this script restating any of them.
set -a
# shellcheck disable=SC1091
source .env
set +a

: "${POSTGRES_USER:?POSTGRES_USER is not set in .env}"
: "${POSTGRES_DB:?POSTGRES_DB is not set in .env}"

# Which database to dump. Defaults to the deployment's, and is overridable so the same script
# can capture a scratch database during a restore rehearsal.
DATABASE="${DATABASE:-$POSTGRES_DB}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DESTINATION="${1:-backups/${DATABASE}-${TIMESTAMP}.dump}"

mkdir -p "$(dirname "$DESTINATION")"

if ! docker compose ps --status running --services | grep -qx postgres; then
  echo "db_backup: the 'postgres' service is not running. Start it with 'docker compose up -d postgres'." >&2
  exit 1
fi

echo "db_backup: dumping '${DATABASE}' to ${DESTINATION}"

# `-T` because there is no terminal on either end of this and the dump is binary: without it
# Docker allocates a TTY and the stream is line-ending mangled into an unrestorable file.
# The dump is written to a temporary name and moved into place only after `pg_dump` succeeds,
# so a failed or interrupted run cannot leave a truncated file that looks like a backup.
TEMPORARY="${DESTINATION}.partial"
trap 'rm -f "$TEMPORARY"' EXIT

docker compose exec -T postgres \
  pg_dump --username "$POSTGRES_USER" --dbname "$DATABASE" --format=custom \
  > "$TEMPORARY"

mv "$TEMPORARY" "$DESTINATION"
trap - EXIT

echo "db_backup: wrote $(du -h "$DESTINATION" | cut -f1) to ${DESTINATION}"
