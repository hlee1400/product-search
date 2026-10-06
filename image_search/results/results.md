Gallery: 23,807 products. Queries: 2,000 (seed 0). One correct answer per query.


### Same model

| Search type | Model | R@1 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| text -> image | CLIP ViT-B/32 | 0.204 | 0.464 | 0.278 |
| image -> text | CLIP ViT-B/32 | 0.148 | 0.356 | 0.205 |
| text -> text | CLIP ViT-B/32 | 0.538 | 0.714 | 0.595 |
| image -> image | CLIP ViT-B/32 | 0.788 | 0.925 | 0.835 |
| text -> image | SigLIP B/16 | 0.622 | 0.880 | 0.708 |
| image -> text | SigLIP B/16 | 0.518 | 0.807 | 0.611 |
| text -> text | SigLIP B/16 | 0.622 | 0.813 | 0.684 |
| image -> image | SigLIP B/16 | 0.895 | 0.967 | 0.921 |
| text -> image | SigLIP 2 B/16 | 0.608 | 0.881 | 0.700 |
| image -> text | SigLIP 2 B/16 | 0.544 | 0.815 | 0.635 |
| text -> text | SigLIP 2 B/16 | 0.618 | 0.806 | 0.680 |
| image -> image | SigLIP 2 B/16 | 0.933 | 0.988 | 0.955 |

### Separate models

| Search type | Model | R@1 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| image -> image | DINOv2 B/14 | 0.986 | 0.997 | 0.990 |
| text -> text | MiniLM-L6 | 0.645 | 0.797 | 0.696 |

### Fused

| Search type | Model | R@1 | R@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| text -> product (RRF) | MiniLM-L6 + CLIP ViT-B/32 | 0.422 | 0.803 | 0.539 |
| text -> product (RRF) | MiniLM-L6 + SigLIP B/16 | 0.686 | 0.905 | 0.756 |
| text -> product (RRF) | MiniLM-L6 + SigLIP 2 B/16 | 0.686 | 0.907 | 0.754 |
