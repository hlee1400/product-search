# Product Search API (Fly.io)

A hosted HTTP API over the image search index: text search, photo search and
"more like this", with color / category filters. The same app also hosts the
search page at `/`, behind a site password ([`web.py`](web.py)). It runs the same encoders and
the same score blend as the benchmarks in [`image_search/`](../image_search/README.md).

## Data storage: what goes where, and why

At query time the API needs four things. Each gets the simplest storage that fits how big
it is and how often it changes:

| What | Size (24k items → 50k) | Changes | Where it lives |
| --- | --- | --- | --- |
| Models: SigLIP 2 (text + photo), MiniLM (text) | ~1.6 GB on disk, ~0.9 GB resident | rarely | **Docker image layer**, downloaded at build time |
| Vectors: SigLIP 2 photo + MiniLM title per item | 55 MB → ~115 MB (float16) | on index rebuild | **Docker image** (`api/bundle/`), held in RAM |
| Item metadata: title, price, URL, color, category | ~5 MB → ~10 MB | on index rebuild | **Docker image** (`api/bundle/items.parquet`), held in RAM |
| Product photos | ~20 KB each, 0.5 → 1 GB | on index rebuild | **Tigris bucket** (Fly's S3-compatible object storage), public URLs |

Why this split:

- **Models in the image.** A machine starts from scratch after every deploy. Downloading
  1.6 GB from Hugging Face on every cold start would be slow and depend on a third party. As
  an image layer below the code, they're built once and cached.
- **The vectors don't need a vector database.** Scoring a query against 50k vectors is one
  matrix multiply of a few milliseconds on a CPU. A vector DB (pgvector, Qdrant) pays off at
  millions of vectors, with frequent writes, or with many writers. Here it would add a network
  hop and another service to run.
- **The index ships inside the image** because it's read-only between rebuilds. Each deploy
  is then one versioned unit (`manifest.json` records the git commit and build time), and
  rolling back the code rolls back the data too. If the catalog later changes often, the
  bundle can move to Tigris and load at startup without code changes.
- **No Fly volume.** Volumes pin a machine to one disk in one region. They're for data the
  app writes, and nothing here is written at query time.
- **Photos go in object storage, not the image.** At ~1 GB they'd make every deploy push
  1 GB, and retailer image links are sometimes dead or block hotlinking. Tigris serves them
  by public URL with long cache headers.
- **When to add a database:** user accounts, saved searches, click or query logs (to tune
  the blend weight on real traffic), or live catalog updates. That's the point to add Fly
  Postgres and, if vectors start changing live, pgvector.

## Machine and cost choices

The final setup is **4 GB RAM, 2 shared CPUs, suspended when idle**. It took four deploys to
get there, and each step fixed what the previous one measured:

| Setup | What I measured | Problem |
| --- | --- | --- |
| 4 GB, `stop` when idle | First request after idle: **HTTP 502 after 45 s** | A cold start reloads and warms the models (~65 s), and Fly's proxy gives up first |
| 2 GB, `suspend` when idle | Text wakes in 3.4 s, but the first **photo** query took **16 s** | The process uses ~1.8 GB (0.95 GB own memory + 0.87 GB of weights paged in from disk). At 2 GB the photo model's weights got evicted and were re-read from Fly's slow disk |
| 2 GB + int8 models | Photo search MRR@10 **0.971 → 0.087**, text −0.02 to −0.05 | Naive 8-bit quantization breaks the vision transformer; never deployed ([`check_quantization.py`](check_quantization.py)) |
| **4 GB, `suspend` when idle** | First photo query after a wake-up **~1.3 s**, then 0.26 s text / 0.43 s photo | Fly's docs discourage suspend above 2 GB; it has worked reliably in testing |

Details:

- **Why suspend, not stop.** Suspend snapshots the machine's memory, with the models already
  loaded and warmed, and resumes it on the next request. Stop throws that away, and the next
  request pays the full ~65 s startup.
- **Each deploy discards the snapshot,** so the first start after a deploy is cold (~65 s, in
  which the first wake-up afterwards was 9.9 s). Every later wake-up takes ~1.3 s.
- **Startup warm-up.** On Fly the first pass through each model took ~20 s (weights are
  memory-mapped from disk and CPU kernels initialize lazily). The server runs one text and
  one photo pass before accepting traffic, so no user pays that.
- **CPU threads.** The server sizes torch's thread pool from the container's CPU quota. In
  Docker with `--cpus 2` on a 16-core host, torch defaulted to 16 threads and every query
  took **18 s** instead of ~0.1 s.
- **Cost.** While suspended, the machine isn't billed for CPU or RAM, only storage. While
  awake, it's billed as a 4 GB shared-CPU machine. For no wake-up delay at all, set
  `min_machines_running = 1` (always on, billed every hour).
- **API key required.** The catalog isn't redistributable, so `/v1/*` routes need
  `Authorization: Bearer <key>` or `X-API-Key: <key>`. The key is a Fly secret, never in the
  repo. The server refuses to start without one (`ALLOW_NO_AUTH=1` is for local development only).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness; no key needed (Fly's health check) |
| GET | `/v1/facets` | color / category counts and index info |
| POST | `/v1/search/text` | `{"query": "...", "mode": "blend"\|"photo"\|"title", "limit": 20, "color": "red", "category": "Apparel & Accessories"}` |
| POST | `/v1/search/image` | multipart: `image` file, optional `limit`, `color`, `category` |
| GET | `/v1/items/{id}` | one item |
| GET | `/v1/items/{id}/similar` | items with similar photos; optional `limit`, `color`, `category` |

`mode: "blend"` (the default) is the tuned search: 70% SigLIP 2 photo score + 30% MiniLM
title score, z-normalized per query. Interactive docs are at `/docs`.

```bash
curl https://<app>.fly.dev/v1/search/text \
  -H "Authorization: Bearer $SEARCH_API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "white star sneakers with green accents", "limit": 5}'
```

## The hosted search page

`/` serves the same page as the local app (`image_search/static/index.html`), limited to the
deployed models: blend (the default), photo-only, title-only, and photo search / "more like
this". The page calls `/api/status` and `/api/search`, which have the same shapes as the
local app's, so one page serves both.

- **Login:** `/login` checks the `SITE_PASSWORD` Fly secret and sets a signed, HttpOnly,
  Secure session cookie (30 days). A wrong password waits 1 s to slow down guessing.
  Changing the password logs everyone out (the cookie key is derived from it).
- **The API key never reaches the browser.** The page's endpoints run searches in-process
  with the session cookie, and `/v1/*` still requires the API key.
- **Same machine, no extra cost.** The page and the API share one process and one index.
- `/about` describes the API; `/docs` is the interactive API reference.

## Live deployment

- Search page: https://hlee-product-search.fly.dev (password in the `SITE_PASSWORD` user
  environment variable on the dev machine)
- App: `hlee-product-search`, at https://hlee-product-search.fly.dev (Chicago, `ord`)
- Photos: public Tigris bucket `hlee-product-photos`
- API key: the `API_KEY` Fly secret. The same key is in the `SEARCH_API_KEY` user
  environment variable on the dev machine.

## Deploying

One-time setup (needs [flyctl](https://fly.io/docs/flyctl/install/) and a Fly account):

```powershell
fly auth login
fly launch --no-deploy --copy-config        # creates the app from fly.toml
fly secrets set API_KEY=<long random key>   # the key clients send
fly secrets set SITE_PASSWORD=<password>    # the search page's login
fly storage create --public                 # photo bucket; prints its credentials
```

Photos (re-run after the index grows; it skips what's already uploaded):

```powershell
$env:AWS_ACCESS_KEY_ID="..."; $env:AWS_SECRET_ACCESS_KEY="..."; $env:BUCKET_NAME="..."
python api/upload_photos.py
# then set PHOTO_BASE_URL in fly.toml to https://<bucket>.fly.storage.tigris.dev
```

Each deploy (after rebuilding the index or changing the code):

```powershell
python api/export_index.py   # writes api/bundle/ from image_search/ artifacts
fly deploy                   # builds api/Dockerfile and rolls out
```

Local test without Fly:

```powershell
docker build -f api/Dockerfile -t product-search-api .
docker run -p 8080:8080 -e API_KEY=dev -e SITE_PASSWORD=dev product-search-api
```
