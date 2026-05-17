# NMR Predictor HPC database setup

A one-pass runbook for standing up the PostgreSQL database that
`wf-db-ingest` writes to at the end of the NMR Predictor pipeline.
Ordered so each step runs cleanly without retries; no chicken-and-egg
loops, no manual edits to `pg_hba.conf`.

Two long-lived environments are involved:

- **`nmrdb`** — PostgreSQL server only. Activated by name:
  `micromamba activate nmrdb`.
- **`/gpfs/group/shenvi/envs/workflow312`** — workflow runtime +
  `nmr-data` Python package. Must be activated by full path:
  `micromamba activate /gpfs/group/shenvi/envs/workflow312`.

All persistent state lives under `/gpfs/group/shenvi/nmr-data/`:

```
/gpfs/group/shenvi/nmr-data/
├── pg/             # Postgres cluster ($PGDATA)
├── pg_sockets/     # unix-socket directory
└── runs/           # NMR_HPC_DATA_ROOT — workflow output staging
```

---

## Phase A — Install (once)

```bash
# 1. Create the Postgres server env
micromamba create -y -n nmrdb -c conda-forge postgresql=16

# 2. Install scripps-workflow + nmr-data into workflow312
micromamba activate /gpfs/group/shenvi/envs/workflow312
cd /gpfs/group/shenvi/code/scripps-workflow

# nmr-data is a private GitHub repo. Make sure the SSH-over-HTTPS
# rewrite is in place so pip can clone it via your existing SSH key.
git config --global url."git@github.com:".insteadOf "https://github.com/"

python -m pip install -e ".[db]"

# Smoke test
python -c "
from nmr_data.db import get_session
from nmr_data.ingest import ingest_nmr_aggregate_result
print('nmr_data imports OK')
"
```

---

## Phase B — Database server (once)

```bash
micromamba activate nmrdb
export PGDATA=/gpfs/group/shenvi/nmr-data/pg
mkdir -p "$PGDATA" \
         /gpfs/group/shenvi/nmr-data/pg_sockets \
         /gpfs/group/shenvi/nmr-data/runs
chmod 700 "$PGDATA"

# Generate strong alphanumeric passwords upfront — no URL-special chars
POSTGRES_PW="$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)"
NMRDATA_PW="$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)"

# Stash both in ~/.pgpass BEFORE running anything that prompts.
# The format is hostname:port:database:username:password.
PGPASS="$HOME/.pgpass"
touch "$PGPASS"; chmod 600 "$PGPASS"
echo "localhost:5433:*:postgres:${POSTGRES_PW}"     >> "$PGPASS"
echo "localhost:5433:nmrdata:nmrdata:${NMRDATA_PW}" >> "$PGPASS"

# Initialize the cluster. --pwfile seeds the postgres superuser
# password during init, sidestepping the chicken-and-egg of needing
# to connect AS postgres before postgres has a password.
PWFILE="$(mktemp)"; printf "%s" "$POSTGRES_PW" > "$PWFILE"
initdb -D "$PGDATA" \
       --username=postgres \
       --pwfile="$PWFILE" \
       --auth-host=scram-sha-256 \
       --auth-local=peer \
       --encoding=UTF8
rm -f "$PWFILE"

# Cluster-local config — non-default port (avoids cluster-wide 5432
# conflicts), localhost-only, daily log rotation.
cat >> "$PGDATA/postgresql.conf" <<'EOF'

# scripps-workflow / nmr-data overrides
listen_addresses = 'localhost'
port = 5433
unix_socket_directories = '/gpfs/group/shenvi/nmr-data/pg_sockets'
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%a.log'
log_rotation_age = 1d
log_truncate_on_rotation = on
EOF

# Start (pg_ctl daemonizes; no tmux needed)
pg_ctl -D "$PGDATA" -l "$PGDATA/server.log" start
pg_ctl -D "$PGDATA" status   # expect: "server is running (PID: ...)"
ss -ltn | grep 5433          # expect a LISTEN on 127.0.0.1:5433

# Forget the passwords from this shell now that they're in ~/.pgpass
unset POSTGRES_PW
```

---

## Phase C — Create the application role + database (once)

`~/.pgpass` now has the postgres superuser password, so this runs
without prompts. The `nmrdata` password is interpolated into the SQL
inside a heredoc so it never lands in shell history.

```bash
# NMRDATA_PW is still set in this shell from Phase B. If you've
# rotated shells, re-read it from ~/.pgpass:
#   NMRDATA_PW="$(awk -F: '$1=="localhost" && $4=="nmrdata" {print $5; exit}' ~/.pgpass)"

psql --host=localhost --port=5433 --username=postgres --dbname=postgres <<SQL
CREATE USER nmrdata WITH LOGIN PASSWORD '${NMRDATA_PW}';
CREATE DATABASE nmrdata WITH OWNER nmrdata ENCODING 'UTF8';
SQL

unset NMRDATA_PW

# Verify — should NOT prompt; libpq finds the password in ~/.pgpass
psql 'postgresql://nmrdata@localhost:5433/nmrdata' -c '\conninfo'
# Expected: "You are connected to database 'nmrdata' as user 'nmrdata'..."
```

---

## Phase D — Activation hooks (once)

Pin the relevant env vars into each env's activation hook so future
sessions just work.

### `nmrdb` (Postgres admin shell)

```bash
ACT=/gpfs/group/shenvi/envs/nmrdb/etc/conda/activate.d
mkdir -p "$ACT"
cat > "$ACT/pgdata.sh" <<'EOF'
export PGDATA=/gpfs/group/shenvi/nmr-data/pg
EOF
chmod 600 "$ACT/pgdata.sh"
```

### `workflow312` (workflow runtime)

```bash
ACT=/gpfs/group/shenvi/envs/workflow312/etc/conda/activate.d
mkdir -p "$ACT"
cat > "$ACT/nmr-data.sh" <<'EOF'
# Password lives in ~/.pgpass — keep this URL credential-less so
# subprocess env, scrollback, and manifests never see a secret.
export NMR_DATABASE_URL='postgresql://nmrdata@localhost:5433/nmrdata'
export NMR_HPC_DATA_ROOT='/gpfs/group/shenvi/nmr-data/runs'
EOF
chmod 600 "$ACT/nmr-data.sh"
```

### Verify

```bash
micromamba deactivate
micromamba activate nmrdb
test -n "$PGDATA" && echo "PGDATA set: $PGDATA"

micromamba deactivate
micromamba activate /gpfs/group/shenvi/envs/workflow312
test -n "$NMR_DATABASE_URL" && echo "NMR_DATABASE_URL set (length ${#NMR_DATABASE_URL})"
test -n "$NMR_HPC_DATA_ROOT" && echo "NMR_HPC_DATA_ROOT set: $NMR_HPC_DATA_ROOT"
```

Do NOT `echo "$NMR_DATABASE_URL"` — if you ever included the password
in the URL by accident, that would print it. The activation hook here
is password-less by design, but the safer habit is to print only the
length or the non-secret parts.

---

## Phase E — Schema migrations (once)

`nmr-data` ships its own schema. Discover the mechanism, then run it:

```bash
micromamba activate /gpfs/group/shenvi/envs/workflow312

# Discovery
NMR_DATA_DIR="$(python -c 'import nmr_data, os; print(os.path.dirname(nmr_data.__file__))')"
echo "nmr_data at: $NMR_DATA_DIR"
find "$NMR_DATA_DIR" -maxdepth 4 \
     \( -name 'alembic.ini' -o -name 'migrations' -type d -o -name 'cli.py' \)
python -c "import nmr_data, pkgutil; print([m.name for m in pkgutil.iter_modules(nmr_data.__path__)])"
```

Based on what discovery reveals, run one of:

### Path 1 — alembic config inside the package

```bash
alembic -c "$NMR_DATA_DIR/alembic.ini" upgrade head
```

### Path 2 — module CLI

```bash
# Substitute the real submodule name if it isn't "cli"
python -m nmr_data.cli migrate
```

### Path 3 — SQLAlchemy fallback (no migration history, just builds schema)

```bash
python <<'PY'
import importlib
from nmr_data.db import engine
for mod in ('nmr_data.models', 'nmr_data', 'nmr_data.db'):
    try:
        m = importlib.import_module(mod)
        Base = (
            getattr(m, 'Base', None)
            or getattr(getattr(m, 'models', None), 'Base', None)
        )
        if Base is not None:
            Base.metadata.create_all(engine)
            print(f'schema created via {mod}.Base.metadata.create_all')
            break
    except Exception as e:
        print(f'  {mod}: {e}')
PY
```

### Confirm

```bash
psql 'postgresql://nmrdata@localhost:5433/nmrdata' -c '\dt'
# Expected: molecules, predicted_runs, predicted_shifts,
#           predicted_couplings, conformers, ...
```

---

## Phase F — End-to-end smoke test (once)

```bash
micromamba activate /gpfs/group/shenvi/envs/workflow312

# 1. DB reachable from the workflow side, no prompts
python -c "
from nmr_data.db import get_session
from sqlalchemy import text
with get_session() as s:
    print(s.execute(text('select current_database(), current_user')).fetchone())
"
# Expected: ('nmrdata', 'nmrdata')

# 2. Ingest function importable
python -c "from nmr_data.ingest import ingest_nmr_aggregate_result; print('ingest OK')"

# 3. Data root writable
test -d "$NMR_HPC_DATA_ROOT" && test -w "$NMR_HPC_DATA_ROOT" && echo 'data root OK'
```

If all three print success, run the NMR Predictor workflow in the GUI
with `dry_run=true` on the `wf_db_ingest` node. Inspect its manifest
for the "would have inserted N molecule rows" log. Flip to
`dry_run=false` once the dry run is clean.

---

## Daily-use cheat sheet

| Task | Command |
|---|---|
| Check Postgres status | `micromamba activate nmrdb && pg_ctl status` |
| Start Postgres (after node reboot) | `micromamba activate nmrdb && pg_ctl -D "$PGDATA" -l "$PGDATA/server.log" start` |
| Stop Postgres cleanly | `micromamba activate nmrdb && pg_ctl -D "$PGDATA" stop -m fast` |
| Open `psql` as the workflow user | `micromamba activate nmrdb && psql 'postgresql://nmrdata@localhost:5433/nmrdata'` |
| Tail server log | ``tail -F "$PGDATA"/log/postgresql-*.log`` |
| Inspect schema | `psql 'postgresql://nmrdata@localhost:5433/nmrdata' -c '\dt'` |
| Count rows in a table | `psql 'postgresql://nmrdata@localhost:5433/nmrdata' -c 'select count(*) from molecules;'` |

---

## Security notes

- **Passwords live in `~/.pgpass` (mode 0600) only.** Never put a DB
  password into an env var, a script committed to git, or anything
  that gets echoed to a terminal. `libpq` reads `~/.pgpass`
  automatically for psql, psycopg, and SQLAlchemy-via-psycopg.
- **The `NMR_DATABASE_URL` env var is password-less** by design:
  `postgresql://nmrdata@localhost:5433/nmrdata`. Combined with
  `~/.pgpass`, that's enough for any client to authenticate.
- **Rotating a password.** Generate a new one with
  `openssl rand -base64 24 | tr -d '/+=' | head -c 32`, edit
  `~/.pgpass` to match, then run
  `ALTER USER nmrdata WITH PASSWORD '<new>';` via a `psql ... <<SQL`
  heredoc so the new password isn't visible in shell history.
- **Locked-down `pg_hba.conf`.** Only scram-sha-256 over localhost.
  Confirm with `grep ^host "$PGDATA/pg_hba.conf"` — if anything says
  `trust`, treat it as a configuration error.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pg_ctl: directory "" does not exist` | `$PGDATA` not set in this shell | `export PGDATA=/gpfs/group/shenvi/nmr-data/pg` or activate `nmrdb` so the hook fires |
| `psql: command not found` | psql lives in `nmrdb`, not `workflow312` | `micromamba run -n nmrdb psql ...` |
| `password authentication failed for user "postgres"` | postgres role has no password | re-run `initdb --pwfile=...` (Phase B) or set the password in single-user mode |
| `Defaulting to user installation because normal site-packages is not writeable` (from pip) | pip is the system pip, not the env's pip | use `python -m pip install ...` instead of bare `pip` |
| `directory "/var/lib/postgresql/data" / "$PGDATA" is not empty` | leftover from a previous initdb | `rm -rf "$PGDATA"/* "$PGDATA"/.*` (be careful — this nukes the cluster) before re-running initdb |
| `Address already in use` on `pg_ctl start` | port 5433 occupied | pick another port, edit `postgresql.conf`, retry |
