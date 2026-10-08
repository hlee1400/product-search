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
| [6](#step-6-add-product-attributes-color-category) | Add color + category: open or closed labels? Do they help ranking? | Closed 1-of-N labels win; boosting by color doesn't help | Color / category as user filters |
| [7](#step-7-host-it-as-an-api) | How to serve it? | Read-only 60 MB index fits in memory; threads, sleep mode and memory each needed fixing | API on Fly.io (4 GB, suspend), index in the image, photos in Tigris |

The detailed log of steps 3–5 is in [IMPROVEMENTS.md](IMPROVEMENTS.md); the API's design is in
[api/README.md](../api/README.md).

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

## Step 6: add product attributes (color, category)

**What I did.** Shoppers filter by color and category, and the catalog has neither, so I had
a vision model label every product through LunaRoute
([`extract_attributes.py`](extract_attributes.py), photo + title in, JSON out):

- **Category:** one of the 192 second-level paths of the
  [Google product taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt),
  e.g. "Apparel & Accessories > Shoes". Level 3 has 1,349 paths, too many to pick from in
  one request. The response schema only allows real paths, so every answer is valid.
- **Color:** a 16-color palette broad enough to filter on (black, white, gray, beige, brown,
  red, pink, orange, yellow, green, blue, purple, gold, silver, multicolor, clear).

There were two ways to get color, so I tried both on a 300-product pilot:

- **Open:** the model names the color freely ("dusty rose"), and a synonym map
  ([`attributes.py`](attributes.py)) normalizes it to the palette.
- **Closed (1-of-N):** the model must pick a palette value.

They agreed on 78%. I checked all 65 disagreements against the photos:

- **Two-color products (~35 cases):** open said "black and white" and normalized to
  multicolor, while closed picked the dominant color. For a black sneaker with a white sole,
  closed is what a shopper filtering by "black" wants.
- **Neutral shades (~12 cases):** cream, camel and taupe. Open was better here, but only
  because the closed prompt hadn't said where beige ends.
- **Open's structural problem:** the synonym map never covers every word ("orchid",
  "berry" came back unmapped).

**What it led to.** The closed method, with the palette boundaries written into the prompt
and a free-text `color_name` returned in the same call, so "dusty rose" survives for display.
All 23,807 products were labeled in ~50 minutes (12.4M input tokens), with no failures. The
app and API filter every search type by color and category.

**Does color also help ranking?** Filters are the obvious use. The less obvious question is
whether a query that names a color ("red summer dress") should boost red products
automatically. I measured it ([`tune_color_boost.py`](tune_color_boost.py)) on the 1,457
benchmark queries that name a color, with the boost tuned on half the products:

| Method (held-out, color queries) | from brand | from image | from text | mean MRR@10 |
| --- | --- | --- | --- | --- |
| **no boost (blend only)** | **0.982** | 0.516 | **0.599** | **0.699** |
| boost 0.25 (best on tune half) | 0.964 | 0.527 | 0.587 | 0.693 |
| boost 1.0 | 0.935 | 0.515 | 0.561 | 0.670 |
| filter to the query's color | 0.714 | 0.450 | 0.461 | 0.542 |

- **No boost helps.** The best one breaks even at most, and hard-filtering to the query's
  color loses a quarter of the score.
- **The reason is visible in the labels.** When the query writer saw the photo, the
  classified color matched the color they named 83% of the time. So labels are mostly right,
  but SigLIP already sees color in the photo, and the boost only adds the ~17% disagreement as
  noise.

**What it led to.** Color stays an **explicit filter the user chooses**. It isn't applied
automatically from query words. Without this test, a color boost would have been the
obvious next feature.

## Step 7: host it as an API

**What I did.** Packaged the final design as an HTTP API on Fly.io ([`api/`](../api/README.md)):
text search with the w = 0.7 blend, photo search, "more like this", and color / category
filters, behind an API key.

**How the data is stored, and why.** The index is read-only between rebuilds and small (60 MB
of vectors and metadata at 24k items, ~115 MB at 50k), so:

- **Vectors and metadata ship inside the Docker image** and are searched in memory. One
  matrix multiply over 50k vectors takes milliseconds, so a vector database would only add a
  network hop.
- **Models are baked into the image,** so a machine waking from idle never downloads from
  Hugging Face.
- **Photos go to Tigris object storage** rather than into the image, so a deploy doesn't push
  1 GB.
- **No database yet.** Nothing is written at query time. User accounts, query logs or live
  catalog updates would be the point to add one.

**What I found while testing it.**

- **Threads.** The first container run took **18 s per query**. Docker capped it at 2 CPUs,
  but torch saw all 16 host cores and started 16 threads that fought over the quota. Sizing
  torch's threads from the container's CPU limit brought it to ~0.1 s.
- **Sleeping when idle.** Stopping the machine made the first request after idle fail (a 502
  after 45 s, since startup takes ~65 s). Suspending it instead keeps the loaded models in
  memory. At 2 GB the photo model's weights were evicted, and the first photo search after a
  wake-up took 16 s.
- **Shrinking the models didn't work.** 8-bit weights would have fit 2 GB, but they dropped
  photo search from 0.971 to 0.087 MRR@10, so I measured that and never deployed it.
- **Final setup:** 4 GB, suspended when idle. Live, it answers text in ~0.26 s and photos in
  ~0.43 s; after a wake-up, the first photo search takes ~1.3 s. The steps are in
  [api/README.md](../api/README.md#machine-and-cost-choices).

## Still open

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
  attributes.py         step 6: color palette, color normalization, Google taxonomy, query colors
  extract_attributes.py step 6: color + category labels via LunaRoute -> attributes/
  tune_color_boost.py   step 6: does boosting the query's color help ranking?
  app.py                Flask API (port 5001)
  static/               search page
  results/              benchmark output
  IMPROVEMENTS.md       detailed log of steps 3-5
```
