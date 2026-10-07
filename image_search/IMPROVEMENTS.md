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
| 1 | Tune a **weighted** combination of title search and photo search | First results below; to re-run on the clean query set |
| 2 | Have **one model (Qwen2.5-VL) write both halves** of the benchmark | Generator ready; waiting on the remote setup |
| 3 | Move query generation **off the laptop**, through LunaRoute | `--backend api` built and tested; needs the LunaRoute/provider details |

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
held-out half, MRR@10 [R@10]):

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

These numbers are **provisional** until step 2 removes the writer confound. If they hold, the
app's fused search should move from equal-weight RRF to the score blend.

## 2. One writer for both halves

Qwen2.5-VL can see images (Gemma can't in this Ollama install), so it can write both halves.
`generate_queries.py` now tracks progress per writer, so a Qwen run doesn't skip products
just because Gemma already covered them. Every query row records its writer, and
`benchmark_queries.py` / `tune_fusion.py --generator <model>` can score one writer's
queries alone.

The plan is to regenerate **both** halves with the same remote Qwen model, rather than adding
remote text queries to the local image ones. The local model is a 4-bit Ollama build and a
hosted one usually isn't, so mixing them would bring back a small version of the same confound.

## 3. Offloading with LunaRoute

**What LunaRoute is.** [LunaRoute](https://deepwiki.com/erans/lunaroute) is a local proxy
for LLM API calls, written in Rust. Clients send it OpenAI- or Anthropic-format requests, and
it forwards them to the configured provider. It adds routing and fallbacks, session recording
and PII redaction, with very little added latency. It doesn't run models itself.

**What that means for this project:**
- **Query generation (the expensive part)** can move off the laptop. With
  `--backend api`, requests go to LunaRoute, which forwards them to a hosted provider running
  Qwen2.5-VL. The laptop only sends requests, so there's no GPU or memory load, and 8 run in
  parallel. LunaRoute adds one place to change provider or model, a record of every request
  for reproducibility, and fallback if a provider fails.
- **The benchmark scoring stays local.** Embedding 3,000 queries and ranking them against the
  23.8k-product index takes about a minute on the GPU, and it needs the local embedding indexes
  anyway. It also isn't an LLM call, so LunaRoute couldn't route it.
- **What's needed to run it:** LunaRoute installed and running, an upstream provider that
  hosts Qwen2.5-VL 7B (e.g. OpenRouter), and that provider's API key in LunaRoute's config.
  Then:

```powershell
python image_search/generate_queries.py --backend api --sources text image `
    --api-model <provider's Qwen2.5-VL 7B name> --api-base-url <LunaRoute URL>/v1
python image_search/benchmark_queries.py
python image_search/tune_fusion.py --generator <provider's Qwen2.5-VL 7B name>
```

`--backend api` works with any OpenAI-compatible endpoint. It was tested end to end against
Ollama's own `/v1` endpoint: parallel requests, the image check and resume all work.

## Log

- **2026-10-07:** Committed the `image-search` findings and created this branch.
  - Added `tune_fusion.py` and ran it on the existing queries (results above).
  - Added `--backend api` and per-writer resume to `generate_queries.py`.
  - LunaRoute: the repo link from search results returns 404 and the docs mirror was
    rate-limited, so the exact install and config steps aren't confirmed yet.
