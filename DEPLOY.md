# Deploying to Vercel

Two services in one project — Next.js frontend (`web/`) + FastAPI backend (`api/`) — declared in the root `vercel.json` via `experimentalServices`.

## One-time provisioning

Run from the repo root.

```bash
# 1. Link the project to Vercel
vercel link

# 2. Add the Marketplace integrations (each opens a tab in the dashboard)
vercel integrations add neon       # Postgres → auto-sets DATABASE_URL
vercel integrations add clerk      # Auth → auto-sets NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY + CLERK_SECRET_KEY

# 3. Set the remaining env vars yourself
vercel env add OLLAMA_BASE_URL     # paste your hosted Ollama URL, e.g. https://ollama.example.com
vercel env add OLLAMA_MODEL        # e.g. gpt-oss:20b
vercel env add CLERK_JWKS_URL      # https://YOUR-DOMAIN.clerk.accounts.dev/.well-known/jwks.json
vercel env add CLERK_ISSUER        # https://YOUR-DOMAIN.clerk.accounts.dev
vercel env add NEXT_PUBLIC_API_URL # the URL of the deployed API service (printed after first deploy)
```

## Required env vars

### Frontend (`web/`)
| Name | Source | Notes |
|---|---|---|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk Marketplace | auto-set |
| `CLERK_SECRET_KEY` | Clerk Marketplace | auto-set |
| `NEXT_PUBLIC_API_URL` | manual | URL of the deployed API service |

### Backend (`api/`)
| Name | Source | Notes |
|---|---|---|
| `DATABASE_URL` | Neon Marketplace | auto-set, e.g. `postgres://…?sslmode=require` |
| `OLLAMA_BASE_URL` | manual | hosted Ollama endpoint; defaults to `http://localhost:11434` for dev |
| `OLLAMA_MODEL` | manual | defaults to `gpt-oss:20b` |
| `CLERK_JWKS_URL` | manual | from Clerk dashboard → API Keys → Public Key. **If unset, the API runs in anonymous mode (dev only).** |
| `CLERK_ISSUER` | manual | matches the dashboard's Frontend API URL |
| `ANTHROPIC_API_KEY` | manual | optional. If set, the API auto-switches to Anthropic and ignores Ollama. |
| `AUTH_DISABLED` | manual | `1` to force anonymous mode in prod (don't). |

## Deploy

```bash
# Preview deploy
vercel

# Production
vercel --prod
```

## Verifying

```bash
# Health
curl https://your-app.vercel.app/api/health

# Routes
curl https://your-app.vercel.app/api/openapi.json | jq '.paths | keys'

# Auth check (should reject without a token in prod)
curl -i https://your-app.vercel.app/api/datasets    # → 401 if Clerk is wired
```

In the browser, sign in with Clerk → drop a CSV → ask a question → confirm tokens stream and a chart appears.

## Architecture notes

- **Persistence**: SQLAlchemy 2 sync. `DATABASE_URL=postgres://…` switches to psycopg2; otherwise SQLite at `api/data.db`.
- **Charts**: PNG bytes stored in the DB (`charts.image_bytes`). Inlined as base64 `data:` URLs in the SSE stream and chat-fetch responses, so `<img>` tags don't need an auth round-trip.
- **Auth**: Every `/api/*` route requires `Authorization: Bearer <Clerk JWT>`. The frontend wires this via `useAuth().getToken()`. JWKS is fetched once per hour and cached.
- **User scoping**: `datasets` and `chats` carry `user_id`. Every read filters by it; every write stamps it.
- **Multi-tenant streaming**: tools.py uses `contextvars` so concurrent requests don't clobber each other's DataFrames.

## Local development

```bash
# Backend (port 8001)
cd api && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --reload --port 8001

# Frontend (port 3000)
pnpm --dir web dev
```

In dev, Clerk runs in keyless mode and the API uses `AUTH_DISABLED` semantics (no `CLERK_JWKS_URL` set → user_id falls back to `"anonymous"`). To exercise the auth path locally, set `CLERK_JWKS_URL` and `CLERK_ISSUER` against a real Clerk dev instance.

## Known limitations to fix before serious traffic

- SQLite is the local default. **Postgres is required for any real deploy.**
- Charts in DB are fine up to a few hundred per chat (~20 KB each). For higher volume, swap to Vercel Blob and store URLs.
- Ollama on Vercel doesn't exist — set `OLLAMA_BASE_URL` to a hosted endpoint, or set `ANTHROPIC_API_KEY` and the provider auto-switches.
- Ephemeral parquet files are written to `api/data/datasets/`. On Vercel Functions this lives in `/tmp` per-invocation. Set `DATA_DIR=/tmp/data` in env so it works across cold starts within the same instance lifetime; for true durability move parquet to Blob too.
