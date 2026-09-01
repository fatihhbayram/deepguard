#!/usr/bin/env bash
#
# Restore a `scripts/db_backup.sh` dump into the Dockerized PostgreSQL (R1-T5).
#
# Usage:
#   scripts/db_restore.sh backups/deepguard-20260901T120000Z.dump
#   scripts/db_restore.sh --database deepguard_scratch backups/....dump
#   scripts/db_restore.sh --yes backups/....dump          # no confirmation prompt
#
# This is the destructive half of the pair, and it is written to say so. `pg_restore --clean`
# drops every object the dump contains before recreating it, so pointing this at the wrong
# database does not merge anything — it replaces it. Three things stand between the operator
# and that:
#
#   - the target database is named back before anything happens, and a confirmation typed as
#     the database's own name is required. `--yes` skips it, for the scripted case;
#   - `--database` restores somewhere other than the deployment's database, which is how a
#     restore is rehearsed without touching the live one;
#   - the worker and the API are not stopped by this script. Restoring underneath running
#     processes is the operator's decision, and 'docker compose stop api api-worker' is the
#     one line that makes it safe. It is deliberately not automated: a script that stops the
#     stack as a side effect of a --database flag pointed at a scratch database would be
#     taking a production outage nobody asked for.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "db_restore: .env not found in $ROOT. Run 'cp .env.example .env' first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${POSTGRES_USER:?POSTGRES_USER is not set in .env}"
: "${POSTGRES_DB:?POSTGRES_DB is not set in .env}"

DATABASE="$POSTGRES_DB"
ASSUME_YES="no"
DUMP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--database)
      DATABASE="${2:?--database needs a database name}"
      shift 2
      ;;
    -y|--yes)
      ASSUME_YES="yes"
      shift
      ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    -*)
      echo "db_restore: unknown option '$1'" >&2
      exit 1
      ;;
    *)
      DUMP="$1"
      shift
      ;;
  esac
done

if [[ -z "$DUMP" ]]; then
  echo "db_restore: no dump file given. Usage: scripts/db_restore.sh [--database NAME] [--yes] <dump>" >&2
  exit 1
fi

if [[ ! -f "$DUMP" ]]; then
  echo "db_restore: '$DUMP' does not exist." >&2
  exit 1
fi

if ! docker compose ps --status running --services | grep -qx postgres; then
  echo "db_restore: the 'postgres' service is not running. Start it with 'docker compose up -d postgres'." >&2
  exit 1
fi

# Refuse a file `pg_restore` cannot read, before the database is touched. `--list` parses the
# dump's table of contents and nothing else, so this costs nothing and turns "restored into an
# emptied database from a corrupt file" into a refusal with the database still intact.
if ! docker compose exec -T postgres pg_restore --list > /dev/null < "$DUMP"; then
  echo "db_restore: '$DUMP' is not a readable custom-format dump." >&2
  exit 1
fi

if [[ "$ASSUME_YES" != "yes" ]]; then
  echo "db_restore: about to restore '${DUMP}' into database '${DATABASE}'."
  echo "            Every object in the dump is DROPPED and recreated there."
  read -r -p "            Type the database name to continue: " CONFIRMATION
  if [[ "$CONFIRMATION" != "$DATABASE" ]]; then
    echo "db_restore: cancelled." >&2
    exit 1
  fi
fi

# Create the target if it is not there yet. This is what makes restoring into a scratch
# database a single command; for the deployment's own database it is a no-op, because the
# database exists. `CREATE DATABASE` has to be issued from another database, hence `postgres`.
if ! docker compose exec -T postgres \
  psql --username "$POSTGRES_USER" --dbname postgres --tuples-only --no-align \
  --command "SELECT 1 FROM pg_database WHERE datname = '${DATABASE}'" | grep -qx 1; then
  echo "db_restore: creating database '${DATABASE}'"
  docker compose exec -T postgres \
    createdb --username "$POSTGRES_USER" "$DATABASE"
fi

echo "db_restore: restoring '${DUMP}' into '${DATABASE}'"

# `--clean --if-exists` so a restore over a populated database replaces it rather than failing
# on every object that already exists, and so a restore into an empty one does not fail on
# every DROP that has nothing to drop.
#
# `--exit-on-error` is deliberate and is the difference between a restore and a hope.
# `pg_restore` defaults to continuing past errors and exiting 0, which means a partially
# restored database reports success — the exact failure a backup exists to prevent.
docker compose exec -T postgres \
  pg_restore --username "$POSTGRES_USER" --dbname "$DATABASE" \
  --clean --if-exists --no-owner --exit-on-error \
  < "$DUMP"

echo "db_restore: restored '${DATABASE}' from ${DUMP}"
