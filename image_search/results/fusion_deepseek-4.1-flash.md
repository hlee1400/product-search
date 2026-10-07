Title model: MiniLM-L6. Photo model: SigLIP 2 B/16. 2,993 queries; weights chosen on 250 products, scored on the other 250.

w is the weight on the photo side (0 = titles only, 1 = photos only).


### Held-out test half: MRR@10 [R@10]

| Method | from image | from text | mean MRR@10 |
| --- | --- | --- | --- |
| title only (MiniLM-L6) | 0.162 [0.29] | 0.438 [0.60] | 0.300 |
| photo only (SigLIP 2 B/16) | 0.599 [0.79] | 0.457 [0.69] | 0.528 |
| equal-weight RRF (current app) | 0.310 [0.52] | 0.511 [0.70] | 0.411 |
| weighted RRF, w=1.0 | 0.600 [0.79] | 0.459 [0.69] | 0.529 |
| score blend, w=0.8 | 0.536 [0.75] | 0.536 [0.73] | 0.536 |

### Weight sweep on the tune half: mean MRR@10

| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rrf | 0.293 | 0.331 | 0.350 | 0.364 | 0.379 | 0.391 | 0.406 | 0.428 | 0.452 | 0.484 | 0.507 |
| blend | 0.293 | 0.326 | 0.364 | 0.409 | 0.450 | 0.483 | 0.508 | 0.534 | 0.545 | 0.537 | 0.507 |
