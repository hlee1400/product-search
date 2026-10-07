# Branch `image-search-improvements`

## Goal

Take the findings from the `image-search` branch ([README](README.md#benchmark-2-llm-written-shopper-queries))
and act on them. That branch compared CLIP, SigLIP, SigLIP 2, DINOv2 and MiniLM on two
benchmarks and found:

1. **Equal-weight RRF fusion is not a safe default.** It was the best text search on queries
   written from product text (MRR@10 0.441), but on queries written from the photo it fell to
   0.211, well below SigLIP 2 photo search alone (0.377).
2. **The query benchmark has a confound.** Gemma wrote the *from text* queries and Qwen2.5-VL
   wrote the *from image* queries, so a gap between the two columns could come from the writer
   rather than the input.
3. **Generating queries locally is slow and strains the machine.** That's about 3 s per
   request on an 8 GB laptop GPU, the vision model doesn't fully fit, and a background server
   was already killed once for low memory.

This branch works on all three:

| # | Change | Status |
| --- | --- | --- |
| 1 | Tune a **weighted** combination of title search and photo search | Done: the app now uses a score blend at w = 0.7 |
| 2 | Have **one model write both halves** of the benchmark | Done: `deepseek-4.1-flash` wrote both (Qwen isn't on LunaRoute) |
| 3 | Move query generation **off the laptop**, through LunaRoute | Done: 1,000 requests in 2 min 10 s, versus over an hour locally |

## 1. Weighted fusion

**Why.** RRF only looks at *ranks*. Rank 1 from title search counts the same whether
MiniLM was confident or guessing. On a visual query like "white star sneakers with green
accents" the title side is mostly guessing, but RRF still gives it half the say. Two fixes
to test:

- **Weighted RRF:** keep ranks, but give the photo side weight `w` and the title side `1 − w`.
- **Score blend:** combine the actual similarity scores, `w · z(photo) + (1 − w) · z(title)`.
  Each side is z-normalized per query (subtract the mean, divide by the standard deviation
  across the gallery) so the two models' score scales are comparable. A query that
  clearly matches one product stands out on that side and carries more weight, so confidence
  is kept.

**How it's measured** ([`tune_fusion.py`](tune_fusion.py)). The weight is swept from 0
(titles only) to 1 (photos only) in steps of 0.1:
- The 500 products are split in half. The weight is **chosen on one half and reported on the
  other**, so the reported number isn't tuned on its own test data.
- The app can't tell whether a query came from text or a photo, so **one weight is chosen for
  both query types**, by their mean MRR@10.

**First results** (Gemma from text + Qwen from image, so the confound still applies;
held-out half, MRR@10 [R@10]; the clean re-run is in [Final results](#final-results-one-writer)):

| Method | from image | from text | mean |
| --- | --- | --- | --- |
| title only (MiniLM) | 0.110 [0.20] | 0.398 [0.54] | 0.254 |
| photo only (SigLIP 2) | 0.373 [0.58] | 0.382 [0.60] | 0.377 |
| equal-weight RRF (current app) | 0.212 [0.36] | 0.445 [0.63] | 0.328 |
| weighted RRF, best w = 1.0 | 0.374 [0.58] | 0.383 [0.60] | 0.379 |
| **score blend, w = 0.8** | 0.348 [0.55] | **0.450 [0.65]** | **0.399** |

What this says so far:
- **Weighted RRF doesn't help.** Its best weight is 1.0, which is just photo search. Every
  amount of title weight costs more on visual queries than it gains on text ones.
- **The score blend does help.** At w = 0.8 it's the best on text-written queries (0.450),
  gives up only a little on image-written ones (0.348 vs 0.373), and has the best mean (0.399
  vs 0.328 for the current app). This supports the reasoning above: keeping score confidence
  matters, and ranks alone throw it away.
- The tune-half sweep is smooth and peaks at 0.7–0.8 (see [results/fusion.md](results/fusion.md)),
  so the choice isn't a lucky spike.

These numbers were **provisional** until step 2 removed the writer confound. They held; see
[Final results](#final-results-one-writer).

## 2. One writer for both halves

The plan was to have Qwen2.5-VL write both halves, since it can see images and Gemma can't in
this Ollama install. But LunaRoute doesn't offer Qwen. What the benchmark needs is that **one
model writes both halves**, not that the model is Qwen. So both halves were regenerated
remotely with **`deepseek-4.1-flash`**, which supports images and JSON output. Using the
remote model for both halves, rather than adding remote text queries to the local Qwen image
queries, also keeps local and hosted versions of a model from mixing.

`generate_queries.py` tracks progress per writer, so a new writer doesn't skip products
another model already covered. Every query row records its writer, and
`benchmark_queries.py` / `tune_fusion.py --generator <model>` can score one writer's queries
alone.

## 3. Offloading with LunaRoute

**What LunaRoute is here.** A hosted, OpenAI-compatible LLM gateway at
`https://gw.lunaroute.com/v1`. `GET /v1/models` lists what it serves: DeepSeek 4.1 Flash, GLM
5.x (including vision variants), and others. Requests use a bearer key, kept in the
`LUNAROUTE_API_KEY` user environment variable and never written to the repo.

**What was offloaded, and what wasn't:**

- **Query generation (the expensive part) moved off the laptop.** `--backend api` (default
  endpoint: LunaRoute, default model: `deepseek-4.1-flash`, `reasoning_effort: none` since the
  task needs no reasoning) sends 8 requests at a time. The laptop only sends requests, with no
  GPU or memory load.

  | | Local (Ollama, 8 GB laptop GPU) | LunaRoute (`deepseek-4.1-flash`) |
  | --- | --- | --- |
  | 1,000 requests (500 products × text + image) | over 1 hour (~3 s each, one at a time) | **2 min 10 s** |
  | Vision | Gemma: broken; Qwen 7B: works | works |
  | Laptop load | GPU full, memory pressure | none |

- **The benchmark scoring stays local.** Embedding a few thousand queries and ranking them
  against the 23.8k-product index takes about a minute on the GPU, and it needs the local
  embedding indexes. It also isn't an LLM call, so the gateway couldn't serve it.

```powershell
# once: [Environment]::SetEnvironmentVariable('LUNAROUTE_API_KEY', '<key>', 'User')
python image_search/generate_queries.py --backend api        # both halves, LunaRoute
python image_search/benchmark_queries.py                     # every writer, side by side
python image_search/tune_fusion.py --generator deepseek-4.1-flash
```

Before an image run, the script still checks that the remote model can see images (it
answered "Red" for the red test square).

## Final results (one writer)

2,993 queries for the same 500 products, all written by `deepseek-4.1-flash`
([results/generated_queries.md](results/generated_queries.md)).

**The writer confound was real, but the conclusions hold.** DeepSeek's queries are sharper
than the local models', so every score is higher. Photo search on image-written queries goes
from 0.377 (Qwen) to 0.594 (DeepSeek), so part of the old gap came from the writer. The
pattern is the same, though:

| MRR@10 | from text | from image |
| --- | --- | --- |
| title search (MiniLM) | 0.443 | 0.150 |
| photo search (SigLIP 2) | 0.441 | **0.594** |
| equal-weight RRF (current app) | **0.516** | 0.309 |

With one writer, the effect of the input is clear: queries written from the photo hardly work
with title search, and equal-weight RRF still falls far below photo search alone on them.

**Fusion tuning** ([results/fusion_deepseek-4.1-flash.md](results/fusion_deepseek-4.1-flash.md);
weight chosen on 250 products, scored on the other 250; MRR@10 [R@10]):

| Method | from image | from text | mean |
| --- | --- | --- | --- |
| title only (MiniLM) | 0.162 [0.29] | 0.438 [0.60] | 0.300 |
| photo only (SigLIP 2) | 0.599 [0.79] | 0.457 [0.69] | 0.528 |
| equal-weight RRF (current app) | 0.310 [0.52] | 0.511 [0.70] | 0.411 |
| weighted RRF, best w = 1.0 | 0.600 [0.79] | 0.459 [0.69] | 0.529 |
| **score blend, w = 0.8** | 0.536 [0.75] | **0.536 [0.73]** | **0.536** |

Conclusions:

- **Replace equal-weight RRF with the score blend at w = 0.8.** The mean MRR@10 goes from 0.411
  to 0.536, and the blend scores the same on both query types instead of failing on one.
- **w = 0.8 was chosen independently on both query sets** (mixed writers and DeepSeek), and
  the tune-half sweep is smooth around it, so it's a stable choice rather than a lucky one.
- **Weighted RRF is a dead end.** On both query sets its best weight is 1.0, which is photo
  search alone.
- **Blend vs photo search alone is close on average** (0.536 vs 0.528). The blend trades some
  accuracy on image-written queries for better text-written ones. It's still the better
  default because these queries leave out brands and model numbers by design, and that's
  exactly where title search is strongest in real traffic.

**Next (done below):** switch the app's fused search to the score blend, and add a brand /
model-number query set so the title side's real value is measured too.

## Brand queries and the final weight

Every query above left out brands and model numbers on purpose, and those are exactly where
title search should be strongest. The blend's small edge over photo search alone (0.536 vs
0.528) might just reflect that bias. So `generate_queries.py --sources brand` adds two
queries per product whose listing shows a brand: brand + product type, and brand + model or
product line. That came to 711 queries for 357 products, via LunaRoute in 50 s.

Held-out products, all three query types, MRR@10 [R@10]
([results/fusion_deepseek-4.1-flash.md](results/fusion_deepseek-4.1-flash.md)):

| Method | from brand | from image | from text | mean |
| --- | --- | --- | --- | --- |
| title only (MiniLM) | 0.772 [0.88] | 0.162 [0.29] | 0.438 [0.60] | 0.457 |
| photo only (SigLIP 2) | 0.594 [0.79] | 0.599 [0.79] | 0.457 [0.69] | 0.550 |
| equal-weight RRF (old app) | 0.759 [0.91] | 0.310 [0.52] | 0.511 [0.70] | 0.527 |
| weighted RRF, best w = 0.9 | 0.678 [0.85] | 0.499 [0.79] | 0.508 [0.71] | 0.562 |
| **score blend, w = 0.7** | **0.822 [0.96]** | 0.495 [0.70] | **0.554 [0.74]** | **0.624** |
| score blend, w = 0.8 | 0.781 [0.94] | 0.536 [0.75] | 0.536 [0.73] | 0.618 |

- **Title search matters on brand queries** (0.772 vs 0.594 for photos), which confirms the
  reason for keeping it.
- **The blend beats both sides on brand queries.** Photos help rank among a brand's many
  similar products.
- **With brand queries in the mix, the tuned weight moves to 0.7.** On held-out products,
  0.7 and 0.8 are nearly tied (0.624 vs 0.618): 0.7 favors brand searches, 0.8 photo-like
  ones. **The app uses 0.7** (`fusion.PHOTO_WEIGHT`), the weight chosen with all three query
  types. Real query logs would settle the exact value.
- **Small leak:** two of the prompt's examples (an Alex and Ani bangle, Golden Goose sneakers)
  happen to be products in the sample. That's 2 of 357 products, too few to move the results.

**App change.** The app's `text → product (fused)` search now blends full-gallery scores
through `fusion.blend` instead of fusing two top-100 lists with RRF. `TXT` / `IMG` tags show
which side alone would have ranked a result in its top 100. Checked in the UI: for "white
star sneakers with green accents", the sneaker with the green heel ranks #1, even though its
title says nothing about green.

## Log

- **2026-10-07:** Committed the `image-search` findings and created this branch.
  - Added `tune_fusion.py` and ran it on the existing queries.
  - Added `--backend api` and per-writer resume to `generate_queries.py`.
  - LunaRoute: the repo link from search results returns 404 and the docs mirror was
    rate-limited, so the install and config steps weren't confirmed.
- **2026-10-07:** Got the LunaRoute gateway details: hosted at `gw.lunaroute.com`,
  OpenAI-compatible, bearer key.
  - Stored the key as the `LUNAROUTE_API_KEY` user environment variable.
  - Listed its models. No Qwen, so `deepseek-4.1-flash` became the single writer.
  - Made LunaRoute the default for `--backend api`, generated 2,993 queries in 2 min 10 s,
    and re-ran the benchmark and fusion tuning (results above).
- **2026-10-07:** Added a brand / model-number query set (`--sources brand`, 711 queries).
  - Re-tuned on all three query types; the best weight moved to 0.7.
  - Moved the blend into `fusion.py` and switched the app's fused search from RRF to it.
  - Rewrote the README as a step-by-step account of how the design changed.
