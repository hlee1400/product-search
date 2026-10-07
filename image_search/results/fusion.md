Title model: MiniLM-L6. Photo model: SigLIP 2 B/16. 3,000 queries; weights chosen on 250 products, scored on the other 250.

w is the weight on the photo side (0 = titles only, 1 = photos only).


### Held-out test half: MRR@10 [R@10]

| Method | from image | from text | mean MRR@10 |
| --- | --- | --- | --- |
| title only (MiniLM-L6) | 0.110 [0.20] | 0.398 [0.54] | 0.254 |
| photo only (SigLIP 2 B/16) | 0.373 [0.58] | 0.382 [0.60] | 0.377 |
| equal-weight RRF (current app) | 0.212 [0.36] | 0.445 [0.63] | 0.328 |
| weighted RRF, w=1.0 | 0.374 [0.58] | 0.383 [0.60] | 0.379 |
| score blend, w=0.8 | 0.348 [0.55] | 0.450 [0.65] | 0.399 |

### Weight sweep on the tune half: mean MRR@10

| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rrf | 0.243 | 0.272 | 0.286 | 0.292 | 0.302 | 0.309 | 0.317 | 0.324 | 0.338 | 0.349 | 0.356 |
| blend | 0.243 | 0.271 | 0.297 | 0.320 | 0.343 | 0.365 | 0.379 | 0.390 | 0.394 | 0.384 | 0.356 |
