Title model: MiniLM-L6. Photo model: SigLIP 2 B/16. 3,704 queries; weights chosen on 250 products, scored on the other 250.

w is the weight on the photo side (0 = titles only, 1 = photos only).


### Held-out test half: MRR@10 [R@10]

| Method | from brand | from image | from text | mean MRR@10 |
| --- | --- | --- | --- | --- |
| title only (MiniLM-L6) | 0.772 [0.88] | 0.162 [0.29] | 0.438 [0.60] | 0.457 |
| photo only (SigLIP 2 B/16) | 0.594 [0.79] | 0.599 [0.79] | 0.457 [0.69] | 0.550 |
| equal-weight RRF (old app) | 0.759 [0.91] | 0.310 [0.52] | 0.511 [0.70] | 0.527 |
| weighted RRF, w=0.9 | 0.678 [0.85] | 0.499 [0.79] | 0.508 [0.71] | 0.562 |
| score blend, w=0.7 (best on tune half) | 0.822 [0.96] | 0.495 [0.70] | 0.554 [0.74] | 0.624 |

### Weight sweep on the tune half: mean MRR@10

| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rrf | 0.459 | 0.492 | 0.504 | 0.512 | 0.519 | 0.521 | 0.528 | 0.536 | 0.542 | 0.552 | 0.527 |
| blend | 0.459 | 0.487 | 0.518 | 0.550 | 0.581 | 0.604 | 0.620 | 0.629 | 0.622 | 0.595 | 0.527 |
