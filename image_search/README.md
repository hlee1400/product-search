# Image Search

Search the product catalog by text or photo, over the 23,807 products that have a photo.

![Comparing search types for "red floral summer dress"](docs/compare.png)

This README describes how the design got to where it is. Each step says what I tried, what
the numbers showed, and what that changed. Where it ends up: text search uses **MiniLM over
titles blended with SigLIP 2 over photos, weighted 70% photo**, combining similarity scores
rather than ranks. A different first guess (equal-weight rank fusion) looked best until the
benchmark got more realistic.

| Step | Question | What I found | What changed |
| --- | --- | --- | --- |
| [1](#step-1-one-shared-model-or-separate-models) | One shared image-text model, or separate models? | SigLIP far beats CLIP; RRF fusion looked best | SigLIP 2 default, RRF for text search |
| [2](#step-2-make-the-queries-realistic) | Do the results hold for realistic queries? | RRF fusion **fails** on queries describing the photo | RRF isn't a safe default |
| [3](#step-3-fix-the-measurement-one-writer-off-the-laptop) | Is that real, or caused by the benchmark? | Partly the benchmark, but the failure is real | One writer, run remotely via LunaRoute |
| [4](#step-4-fix-the-fusion) | Can weighting the fusion fix it? | Weighting ranks doesn't help; blending scores does | Score blend |
| [5](#step-5-check-what-the-blend-might-cost) | Does the blend hurt brand searches? | No, it's best on them too; weight shifts to 0.7 | **App uses score blend, w = 0.7** |

The detailed log of steps 3–5 is in [IMPROVEMENTS.md](IMPROVEMENTS.md).

---

## Step 1: one shared model, or separate models?

**What I did.** I compared two designs for searching across text and images:

- **Same model:** one model embeds images and text into one space, so text can search photos
  directly. I tried three: CLIP ViT-B/32, SigLIP B/16 and SigLIP 2 B/16.
- **Separate models:** a text-only model (MiniLM, as in `search_app`) and an image-only model
  (DINOv2), each searching its own modality.

That gives six search types:

| # | Search type | Design | Model(s) |
| --- | --- | --- | --- |
| 1 | text → image | same model | CLIP, SigLIP, SigLIP 2 |
| 2 | image → text | same model | same three |
| 3 | text → text | same model (its text tower) | same three |
| 4 | image → image | same model (its image tower) | same three |
| 5 | text → text | separate models | MiniLM-L6 |
| 6 | image → image | separate models | DINOv2 B/14 |

Separate models can't search across modalities on their own. The closest they get is a
**fused** text search: MiniLM over titles plus a shared model over photos, merged with
reciprocal rank fusion (RRF), the same method `search_app` uses.

**How I measured it.** The catalog has no human relevance labels, so each query has one right
answer, the product it came from ([`evaluate.py`](evaluate.py), 2,000 queries):

- **Text queries** were the catalog's AI-written product descriptions.
- **Image queries** were cropped, flipped and color-shifted copies of the product photo.

**What I found** (MRR@10; R@1 and R@10 in [results/results.md](results/results.md)):

| Search type | CLIP B/32 | SigLIP B/16 | SigLIP 2 B/16 | Separate model |
| --- | --- | --- | --- | --- |
| text → image | 0.278 | **0.708** | 0.700 | n/a |
| image → text | 0.205 | 0.611 | **0.635** | n/a |
| text → text | 0.595 | 0.684 | 0.680 | **0.696** (MiniLM) |
| image → image | 0.835 | 0.921 | 0.955 | **0.990** (DINOv2) |
| text → product, RRF with MiniLM | 0.539 | **0.756** | 0.754 | n/a |

- **SigLIP is far ahead of CLIP B/32 across modalities** (text → image R@1 0.62 vs 0.20).
- **In each modality, a dedicated model beats the shared one:** DINOv2 on image → image,
  MiniLM on text → text.
- **RRF fusion looked best for text search** (R@10 0.907), because titles and photos fail on
  different products.

**What it led to.** SigLIP 2 as the default shared model, and RRF fusion for text search.

**Why I didn't stop there.** The test queries didn't look like real searches. The AI
descriptions are long, detailed paragraphs, and they mention colors, so they were probably
written while looking at the photo, which flatters photo search. The image → image test only
checks near-duplicate matching.

## Step 2: make the queries realistic

**What I did.** [`generate_queries.py`](generate_queries.py) has an LLM write, for 500 sampled
products, the queries a shopper would actually type, in three styles: **short** ("white chalk
paint"), **specific** ("matte white chalk paint for furniture") and **need** ("paint to give an
old dresser a vintage look"). No brands or model numbers, so the test measures meaning rather
than keyword matches. It writes them twice per product, to see whether the kind of query
matters:

- **from text:** the writer sees only the title, store and store description
- **from image:** the writer sees only the photo

I generated these locally with Ollama. Gemma couldn't actually see images in my Ollama install
(it called a red square "black"), so I added a vision check to the script. Gemma wrote the
text half and Qwen2.5-VL 7B wrote the image half.
[`benchmark_queries.py`](benchmark_queries.py) scores every search type that takes text.

**What I found** (MRR@10):

| Search type | From text (Gemma) | From image (Qwen) |
| --- | --- | --- |
| title search: MiniLM | 0.395 | 0.103 |
| photo search: SigLIP 2 | 0.356 | **0.377** |
| RRF fusion: MiniLM + SigLIP 2 | **0.441** | 0.211 |

- **Title search collapses on queries describing the photo.** Queries like "white star
  sneakers with green accents" describe what's visible, and titles rarely say that.
- **RRF fusion fails on those queries.** At 0.211 it's well below photo search alone. RRF gives
  both lists the same say, so title search's poor guesses push the right product down.
- **Photo search is the steadiest,** and CLIP is behind everywhere.

**What it led to.** Equal-weight RRF, the step 1 winner, isn't a safe default.

**Why I didn't trust it yet.** Two different models wrote the two columns, so part of the gap
could come from the writer rather than the input. Also, generating locally took over an hour
and ran my laptop out of memory once.

## Step 3: fix the measurement (one writer, off the laptop)

**What I did.**

- **One writer.** I regenerated both halves with a single model, `deepseek-4.1-flash`, so the
  writer is the same for both and only the input differs.
- **Off the laptop.** Generation runs through **LunaRoute**, a hosted, OpenAI-compatible LLM
  gateway. The generator's `--backend api` sends 8 requests at a time, and the model runs
  remotely.
- **Scoring stays local.** It takes about a minute on the GPU and needs the local search
  indexes.

| | Local (Ollama, 8 GB laptop GPU) | LunaRoute (`deepseek-4.1-flash`) |
| --- | --- | --- |
| 1,000 requests | over 1 hour | **2 min 10 s** |
| Vision | Gemma broken, Qwen works | works |
| Laptop load | GPU full, memory pressure | none |

**What I found** (MRR@10, one writer):

| Search type | from text | from image |
| --- | --- | --- |
| title search: MiniLM | 0.443 | 0.150 |
| photo search: SigLIP 2 | 0.441 | **0.594** |
| RRF fusion | **0.516** | 0.309 |

- **The writer did matter.** DeepSeek writes sharper queries, so every score rose. Photo search
  on photo-based queries went from 0.377 to 0.594.
- **The pattern held.** With the writer fixed, the input is clearly what matters: title search
  still collapses on photo-based queries, and RRF fusion still falls far below photo search
  alone on them.

**What it led to.** The fusion problem is real, so fix the fusion.

## Step 4: fix the fusion

**Why RRF fails.** RRF only looks at ranks. Rank 1 from title search counts the same whether
MiniLM was confident or guessing. I tried two fixes ([`tune_fusion.py`](tune_fusion.py)):

- **Weighted RRF:** still ranks, but give the photo side weight `w` and the title side `1 − w`.
- **Score blend:** combine the similarity scores, `w · z(photo) + (1 − w) · z(title)`. Each
  side is normalized per query (subtract the mean, divide by the standard deviation across the
  gallery) so the two models' scales are comparable. A confident match stands far above its
  side's average, so confidence carries through.

To avoid fooling myself, the weight is **chosen on half the products and reported on the other
half**. One weight has to serve both query types, because the app can't tell which kind it got.

**What I found** (held-out products, one writer, MRR@10):

| Method | from image | from text | mean |
| --- | --- | --- | --- |
| photo only (SigLIP 2) | 0.599 | 0.457 | 0.528 |
| equal-weight RRF (old app) | 0.310 | 0.511 | 0.411 |
| weighted RRF, best weight | 0.600 | 0.459 | 0.529 |
| **score blend, w = 0.8** | 0.536 | **0.536** | **0.536** |

- **Weighted RRF is a dead end.** Its best weight is 1.0, which is photo search alone. With
  ranks only, any title weight costs more on visual queries than it gains elsewhere.
- **The score blend works.** It scores the same on both query types instead of failing on one,
  and it beats the old app's mean by 0.125.

**What it led to.** The score blend. One open question: on average it's only slightly ahead of
photo search alone (0.536 vs 0.528). So is the title side worth keeping?

## Step 5: check what the blend might cost

**What I did.** Every query so far left out brands and model numbers by design, which are
exactly the searches title matching is built for. So I added a **brand** query set
(`--sources brand`). For each product whose listing shows a brand, the writer adds a
brand + product-type query ("alex and ani charm bangle") and an exact brand + model query
("golden goose super-star sneakers"). That came to 711 queries for 357 products, again via
LunaRoute, in 50 seconds.

**What I found** (held-out products, MRR@10 [R@10]):

| Method | from brand | from image | from text | mean |
| --- | --- | --- | --- | --- |
| title only (MiniLM) | 0.772 [0.88] | 0.162 [0.29] | 0.438 [0.60] | 0.457 |
| photo only (SigLIP 2) | 0.594 [0.79] | 0.599 [0.79] | 0.457 [0.69] | 0.550 |
| equal-weight RRF (old app) | 0.759 [0.91] | 0.310 [0.52] | 0.511 [0.70] | 0.527 |
| **score blend, w = 0.7** | **0.822 [0.96]** | 0.495 [0.70] | **0.554 [0.74]** | **0.624** |
| score blend, w = 0.8 | 0.781 [0.94] | 0.536 [0.75] | 0.536 [0.73] | 0.618 |

- **Title search earns its place on brand queries** (0.772 vs 0.594 for photos), so dropping it
  would have been a mistake.
- **The blend beats both sides on brand queries** (0.822). Even when the title side is right,
  the photo side helps rank among a brand's many similar products.
- **The best weight shifted from 0.8 to 0.7.** On held-out products the two are nearly tied
  (0.624 vs 0.618): w = 0.7 is better for brand searches, w = 0.8 for photo-like ones.

**Where it stands (point B).** The app's text search uses the score blend with **w = 0.7**
([`fusion.py`](fusion.py)), chosen with all three query types included. Against the starting
point, equal-weight RRF, the mean MRR@10 goes from **0.527 to 0.624** on held-out products, and
there's no longer a query type where it collapses.

**Still open:**

- **The right weight depends on real traffic.** If most searches name brands, lower w; if they
  describe looks, raise it. Real query logs would settle it.
- **"Need" queries are weak everywhere** ("bracelet to wear with casual outfits"). The
  cross-encoder reranker from `search_app` is the next thing to try there.
- **"More like this" finding different products in the same style** is still only checked by
  eye.

---

## Running it

Needs the catalog corpus from `search_app` first (`python search_app/build_index.py`).

```powershell
pip install transformers sentencepiece pillow torch sentence-transformers flask pandas pyarrow
python image_search/build_index.py   # embeds 23.8k photos + titles with 5 models, ~15 min on GPU
python image_search/evaluate.py      # step 1 benchmark -> results/results.md
python image_search/app.py           # open http://127.0.0.1:5001
```

Re-running the query benchmarks (steps 2–5):

```powershell
# once: [Environment]::SetEnvironmentVariable('LUNAROUTE_API_KEY', '<key>', 'User')
python image_search/generate_queries.py --backend api --sources text image brand   # via LunaRoute
python image_search/benchmark_queries.py                                           # every writer
python image_search/tune_fusion.py --generator deepseek-4.1-flash                  # fusion weights
```

The generator also supports `--backend ollama` (local; it checks first that the model can see
images) and `--backend claude` (Claude Batch API, needs `ANTHROPIC_API_KEY`).

## The UI

- Type a description, **drop / paste / upload a photo**, or click **More like this** on any
  result to search with that product's photo.
- Search-type chips switch between the search types (only the ones that fit the query appear).
  **Compare all** shows the top 8 of each type side by side.
- **Shared model** picks CLIP, SigLIP, or SigLIP 2 for the same-model and fused types.
- On fused results, `TXT` / `IMG` tags show which side alone would have ranked that product in
  its top 100.
- `?q=...&type=t2p-fused&compare=1&model=siglip2-b16` in the URL runs a search on page load.

## Files

```
image_search/
  encoders.py           model registry + text/image embedding (CLIP, SigLIP, DINOv2, MiniLM)
  build_index.py        gallery + cached embeddings in artifacts/ (git-ignored)
  evaluate.py           step 1: the six search types, R@1 / R@10 / MRR@10
  generate_queries.py   steps 2-5: LLM-written queries (LunaRoute, Ollama or Claude) -> queries/
  benchmark_queries.py  scores the text-query search types on those queries
  tune_fusion.py        steps 4-5: weighted RRF vs score blend, tuned on held-out products
  fusion.py             the score blend the app uses (PHOTO_WEIGHT = 0.7)
  app.py                Flask API (port 5001)
  static/               search page
  results/              benchmark output
  IMPROVEMENTS.md       detailed log of steps 3-5
```
