# Deploy

Two services, two hosts:
- **Web (Next.js)** → Vercel.
- **API (FastAPI)** → Fly.io. (Vercel's Python serverless setup is awkward
  with our package layout; Fly is well-trodden and the repo already has a
  Dockerfile.)

They communicate via plain HTTPS. Frontend points at the API URL via
`NEXT_PUBLIC_API_URL`. Auth is Clerk-issued JWTs sent on every API call.

---

## Current state

✅ **Web is deployed**: https://web-one-henna-87.vercel.app

The Next.js frontend is live in keyless Clerk mode (no real auth — fine
for browsing the landing). To finish, provision the rest:

1. **Postgres** (Neon via Vercel Marketplace) → backs both services
2. **Clerk app** (real keys; replaces keyless mode)
3. **Fly.io app** for the API (`api/Dockerfile` + `api/fly.toml`)
4. **Hosted Ollama URL** wired into the Fly.io app

---

## Web — Vercel

Already linked from `web/` directory. Production deploy:

```bash
cd web
vercel --prod
```

### Required env vars
```bash
vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY  # from Clerk dashboard
vercel env add CLERK_SECRET_KEY                   # from Clerk dashboard
vercel env add NEXT_PUBLIC_API_URL                # the deployed Fly.io URL, e.g. https://data-analyst-chatbot-api.fly.dev
```

Best path for Clerk: `vercel integrations add clerk` opens the Marketplace
and auto-populates the two Clerk env vars.

---

## API — Fly.io

```bash
cd api

# 1. Create the app (only first time)
fly launch --no-deploy --copy-config --name data-analyst-chatbot-api

# 2. Add a persistent volume for parquet/datasets (matches DATA_DIR=/data)
fly volumes create data_volume --region fra --size 1

# 3. Set secrets
fly secrets set DATABASE_URL='postgres://...sslmode=require'      # from Neon
fly secrets set OLLAMA_BASE_URL='https://your-hosted-ollama.com'
fly secrets set OLLAMA_MODEL='gpt-oss:20b'                        # or whichever you host
fly secrets set CLERK_JWKS_URL='https://YOUR-DOMAIN.clerk.accounts.dev/.well-known/jwks.json'
fly secrets set CLERK_ISSUER='https://YOUR-DOMAIN.clerk.accounts.dev'
# Optional — falls back to Ollama if unset:
# fly secrets set ANTHROPIC_API_KEY='sk-ant-...'

# 4. Deploy
fly deploy
```

After deploy, copy the URL Fly prints and set it as `NEXT_PUBLIC_API_URL`
in the Vercel project. Then re-deploy the web service so it picks up the
new env var.

### Health check
```bash
curl https://data-analyst-chatbot-api.fly.dev/health     # → {"status":"ok"}
curl -i https://data-analyst-chatbot-api.fly.dev/datasets # → 401 if Clerk is wired
```

---

## Postgres — Neon

Easiest: `vercel integrations add neon` from the `web/` directory. The
integration auto-sets `DATABASE_URL` for the Next.js project. Copy that
URL value and add it to Fly.io as a secret too:

```bash
fly secrets set DATABASE_URL="$(vercel env pull --output - .env.production | grep DATABASE_URL | cut -d= -f2-)"
```

Or just paste it manually.

---

## Clerk — real keys

If you used the Vercel Marketplace integration above, the publishable key
and secret are already set in `web/`. For the API to verify JWTs you also
need:

- `CLERK_JWKS_URL` — from the Clerk dashboard → API Keys → Public Key
  (the JWKS URL is shown there)
- `CLERK_ISSUER` — same place; it's the Frontend API URL

Set both as Fly.io secrets (see above).

---

## Required env vars summary

### Web (Vercel)
| Name | Source |
|---|---|
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Clerk Marketplace |
| `CLERK_SECRET_KEY` | Clerk Marketplace |
| `NEXT_PUBLIC_API_URL` | Fly.io URL (set after first deploy) |

### API (Fly.io)
| Name | Source |
|---|---|
| `DATABASE_URL` | Neon |
| `OLLAMA_BASE_URL` | your hosted Ollama provider |
| `OLLAMA_MODEL` | model name, e.g. `gpt-oss:20b` |
| `CLERK_JWKS_URL` | Clerk dashboard |
| `CLERK_ISSUER` | Clerk dashboard |
| `ANTHROPIC_API_KEY` | optional — auto-overrides Ollama if set |
| `AUTH_DISABLED` | `1` for anonymous mode (don't use in prod) |

---

## Architecture notes

- **Persistence**: SQLAlchemy 2 sync. `DATABASE_URL=postgres://…` switches
  to psycopg2; otherwise SQLite at `api/data.db`.
- **Charts**: PNG bytes stored in the DB (`charts.image_bytes`). Inlined
  as base64 `data:` URLs in SSE events and chat-fetch responses, so
  `<img>` tags don't need an auth round-trip.
- **Auth**: Every `/api/*` route requires `Authorization: Bearer <Clerk
  JWT>`. The frontend wires this via `useAuth().getToken()`. JWKS is
  fetched once per hour and cached in memory.
- **User scoping**: `datasets` and `chats` carry `user_id`. Every read
  filters by it; every write stamps it.
- **Multi-tenant streaming**: `tools.py` uses `contextvars` so concurrent
  requests don't clobber each other's DataFrames.

## Local development

```bash
# Backend (port 8001)
cd api && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --reload --port 8001

# Frontend (port 3000)
pnpm --dir web dev
```

In dev, Clerk runs in keyless mode and the API uses anonymous mode (no
`CLERK_JWKS_URL` set → user_id falls back to `"anonymous"`). To exercise
the auth path locally, set `CLERK_JWKS_URL` and `CLERK_ISSUER` against a
real Clerk dev instance.
