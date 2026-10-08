Color queries by `deepseek-4.1-flash`: 1,457 of 3,704 queries name a color (brand: 63, image: 970, text: 424). Base score: blend with w = 0.7.

**Label check:** the target product's classified color matches the color the query names in 62% of *from brand* queries, 83% of *from image* queries, 71% of *from text* queries.

Best on the tune half: **boost 0.25**.


### Held-out test half, color queries only: MRR@10 [R@10]

| Method | from brand | from image | from text | mean MRR@10 |
| --- | --- | --- | --- | --- |
| boost 0.0 | 0.982 [1.00] | 0.516 [0.72] | 0.599 [0.83] | 0.699 |
| boost 0.25 **(best on tune)** | 0.964 [1.00] | 0.527 [0.74] | 0.587 [0.82] | 0.693 |
| boost 0.5 | 0.946 [1.00] | 0.528 [0.74] | 0.584 [0.82] | 0.686 |
| boost 0.75 | 0.938 [1.00] | 0.522 [0.75] | 0.577 [0.81] | 0.679 |
| boost 1.0 | 0.935 [1.00] | 0.515 [0.75] | 0.561 [0.79] | 0.670 |
| boost 1.5 | 0.911 [0.96] | 0.501 [0.72] | 0.532 [0.77] | 0.648 |
| boost 2.0 | 0.893 [0.96] | 0.484 [0.71] | 0.507 [0.72] | 0.628 |
| boost 3.0 | 0.860 [0.93] | 0.464 [0.67] | 0.483 [0.68] | 0.602 |
| filter (only matching color) | 0.714 [0.71] | 0.450 [0.64] | 0.461 [0.63] | 0.542 |
