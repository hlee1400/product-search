5,993 LLM-written queries for 500 products. One correct answer per query (the product it was written for).


### MRR@10

| Search type | Model | deepseek-4.1-flash, from text | deepseek-4.1-flash, from image | gemma4:e4b, from text | qwen2.5vl:7b, from image |
| --- | --- | --- | --- | --- | --- |
| text -> text | CLIP ViT-B/32 | 0.322 | 0.122 | 0.244 | 0.074 |
| text -> text | SigLIP B/16 | 0.334 | 0.140 | 0.231 | 0.071 |
| text -> text | SigLIP 2 B/16 | 0.414 | 0.191 | 0.313 | 0.099 |
| text -> text | MiniLM-L6 | 0.443 | 0.150 | 0.395 | 0.103 |
| text -> image | CLIP ViT-B/32 | 0.160 | 0.294 | 0.127 | 0.216 |
| text -> image | SigLIP B/16 | 0.427 | 0.567 | 0.340 | 0.367 |
| text -> image | SigLIP 2 B/16 | 0.441 | 0.594 | 0.356 | 0.377 |
| text -> product (RRF) | MiniLM-L6 + CLIP ViT-B/32 | 0.350 | 0.269 | 0.305 | 0.188 |
| text -> product (RRF) | MiniLM-L6 + SigLIP B/16 | 0.510 | 0.306 | 0.432 | 0.208 |
| text -> product (RRF) | MiniLM-L6 + SigLIP 2 B/16 | 0.516 | 0.309 | 0.441 | 0.211 |

### R@10

| Search type | Model | deepseek-4.1-flash, from text | deepseek-4.1-flash, from image | gemma4:e4b, from text | qwen2.5vl:7b, from image |
| --- | --- | --- | --- | --- | --- |
| text -> text | CLIP ViT-B/32 | 0.457 | 0.215 | 0.368 | 0.141 |
| text -> text | SigLIP B/16 | 0.461 | 0.247 | 0.344 | 0.137 |
| text -> text | SigLIP 2 B/16 | 0.574 | 0.322 | 0.465 | 0.184 |
| text -> text | MiniLM-L6 | 0.588 | 0.262 | 0.528 | 0.183 |
| text -> image | CLIP ViT-B/32 | 0.315 | 0.532 | 0.259 | 0.396 |
| text -> image | SigLIP B/16 | 0.649 | 0.790 | 0.559 | 0.565 |
| text -> image | SigLIP 2 B/16 | 0.680 | 0.797 | 0.587 | 0.574 |
| text -> product (RRF) | MiniLM-L6 + CLIP ViT-B/32 | 0.584 | 0.493 | 0.529 | 0.370 |
| text -> product (RRF) | MiniLM-L6 + SigLIP B/16 | 0.718 | 0.585 | 0.617 | 0.411 |
| text -> product (RRF) | MiniLM-L6 + SigLIP 2 B/16 | 0.711 | 0.592 | 0.625 | 0.407 |

### MRR@10 by query style

| Search type | Model | deepseek-4.1-flash, from text, short | deepseek-4.1-flash, from text, specific | deepseek-4.1-flash, from text, need | deepseek-4.1-flash, from image, short | deepseek-4.1-flash, from image, specific | deepseek-4.1-flash, from image, need | gemma4:e4b, from text, short | gemma4:e4b, from text, specific | gemma4:e4b, from text, need | qwen2.5vl:7b, from image, short | qwen2.5vl:7b, from image, specific | qwen2.5vl:7b, from image, need |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text -> text | CLIP ViT-B/32 | 0.317 | 0.510 | 0.140 | 0.103 | 0.180 | 0.082 | 0.297 | 0.365 | 0.070 | 0.072 | 0.125 | 0.024 |
| text -> text | SigLIP B/16 | 0.279 | 0.561 | 0.162 | 0.094 | 0.229 | 0.096 | 0.256 | 0.379 | 0.059 | 0.058 | 0.124 | 0.030 |
| text -> text | SigLIP 2 B/16 | 0.401 | 0.614 | 0.227 | 0.161 | 0.266 | 0.147 | 0.368 | 0.468 | 0.104 | 0.089 | 0.162 | 0.046 |
| text -> text | MiniLM-L6 | 0.491 | 0.585 | 0.253 | 0.167 | 0.167 | 0.115 | 0.479 | 0.537 | 0.168 | 0.117 | 0.142 | 0.049 |
| text -> image | CLIP ViT-B/32 | 0.148 | 0.231 | 0.101 | 0.263 | 0.397 | 0.222 | 0.157 | 0.158 | 0.066 | 0.213 | 0.338 | 0.097 |
| text -> image | SigLIP B/16 | 0.441 | 0.573 | 0.266 | 0.536 | 0.750 | 0.416 | 0.411 | 0.448 | 0.161 | 0.394 | 0.583 | 0.125 |
| text -> image | SigLIP 2 B/16 | 0.431 | 0.601 | 0.292 | 0.553 | 0.779 | 0.449 | 0.413 | 0.463 | 0.192 | 0.407 | 0.585 | 0.139 |
| text -> product (RRF) | MiniLM-L6 + CLIP ViT-B/32 | 0.372 | 0.461 | 0.216 | 0.273 | 0.341 | 0.194 | 0.374 | 0.387 | 0.153 | 0.205 | 0.280 | 0.080 |
| text -> product (RRF) | MiniLM-L6 + SigLIP B/16 | 0.563 | 0.641 | 0.327 | 0.306 | 0.381 | 0.231 | 0.537 | 0.558 | 0.202 | 0.233 | 0.306 | 0.084 |
| text -> product (RRF) | MiniLM-L6 + SigLIP 2 B/16 | 0.560 | 0.663 | 0.326 | 0.314 | 0.377 | 0.237 | 0.535 | 0.574 | 0.214 | 0.234 | 0.306 | 0.094 |
