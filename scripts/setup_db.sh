#!/usr/bin/env bash
#
# Bring up the local database: Homebrew PostgreSQL 18 + pgvector on port 5433,
# with the `diu` role, the `diu_admission` database and the schema applied.
#
# Port 5433 rather than the default 5432 so that an existing PostgreSQL on 5432
# -- an EnterpriseDB install, for instance -- is left completely alone. It also
# means DATABASE_URL keeps the value it has always had, so no application code
# or config has to change for the database to live here.
#
# Safe to re-run: every step checks before it acts. It never drops anything.
#
#     bash scripts/setup_db.sh
#
set -euo pipefail

PORT=5433
DB=diu_admission
ROLE=diu
PASSWORD=diu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install it from https://brew.sh, then re-run." >&2
    exit 1
fi

if ! brew list postgresql@18 >/dev/null 2>&1; then
    echo "postgresql@18 is not installed. Run:" >&2
    echo "    brew install postgresql@18 pgvector" >&2
    exit 1
fi

PREFIX="$(brew --prefix postgresql@18)"
PGDATA="$(brew --prefix)/var/postgresql@18"
export PATH="$PREFIX/bin:$PATH"

# --- 1. cluster ------------------------------------------------------------
if [ ! -d "$PGDATA" ]; then
    echo "==> initialising a new cluster at $PGDATA"
    initdb --locale=C -E UTF-8 "$PGDATA"
fi

CONF="$PGDATA/postgresql.conf"
if grep -qE '^[[:space:]]*port[[:space:]]*=' "$CONF"; then
    sed -i '' -E "s/^[[:space:]]*port[[:space:]]*=.*/port = $PORT/" "$CONF"
else
    printf '\n# set by scripts/setup_db.sh\nport = %s\n' "$PORT" >>"$CONF"
fi
echo "==> port pinned to $PORT"

# --- 2. server -------------------------------------------------------------
brew services start postgresql@18 >/dev/null

# The service returns before the server is accepting connections, so every
# command below would race it without this wait.
echo -n "==> waiting for the server"
for _ in $(seq 1 30); do
    if pg_isready -h localhost -p "$PORT" -q; then
        echo " -- up"
        break
    fi
    echo -n "."
    sleep 1
done
pg_isready -h localhost -p "$PORT" -q || {
    echo
    echo "server did not come up. Logs:  brew services info postgresql@18" >&2
    exit 1
}

psql_super() { psql -h localhost -p "$PORT" -d postgres -v ON_ERROR_STOP=1 "$@"; }

# --- 3. role and database --------------------------------------------------
# SUPERUSER because CREATE EXTENSION vector requires it -- pgvector is not a
# trusted extension. This is a local development database whose password is
# literally "diu"; it is not an access-control decision.
if [ "$(psql_super -tAc "SELECT 1 FROM pg_roles WHERE rolname = '$ROLE'")" != "1" ]; then
    psql_super -c "CREATE ROLE $ROLE LOGIN SUPERUSER PASSWORD '$PASSWORD'"
    echo "==> created role $ROLE"
fi

if [ "$(psql_super -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB'")" != "1" ]; then
    createdb -h localhost -p "$PORT" -O "$ROLE" "$DB"
    echo "==> created database $DB"
fi

# --- 4. extension and schema ----------------------------------------------
psql -h localhost -p "$PORT" -d "$DB" -v ON_ERROR_STOP=1 \
    -c "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null
echo "==> pgvector $(psql -h localhost -p "$PORT" -d "$DB" -tAc \
    "SELECT extversion FROM pg_extension WHERE extname = 'vector'") ready"

psql -h localhost -p "$PORT" -d "$DB" -U "$ROLE" -v ON_ERROR_STOP=1 -q \
    -f "$REPO_ROOT/migrations/001_schema.sql"
echo "==> schema applied"

echo
echo "Database ready:  postgresql://$ROLE:$PASSWORD@localhost:$PORT/$DB"
echo "Stop it with:    brew services stop postgresql@18"
