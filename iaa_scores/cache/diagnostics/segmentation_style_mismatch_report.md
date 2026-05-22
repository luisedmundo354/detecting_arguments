# Same-Label Segmentation Style Mismatch Diagnostic

Source file: `iaa_scores/cache/diagnostics/segmentation_style_mismatch.json`

Configuration:

| Setting | Value |
|---|---|
| Label mode | `same-label` |
| Overlap threshold | `0.5` |
| Primary edge context | `edge-union` |
| All-pairs sensitivity included | `true` |

## Method

This diagnostic compares the existing one-to-one edit-distance pairing baseline against a new relaxed offset-overlap grouping. The baseline rows labeled `1:1 label-restricted edit` come from the existing edit-distance pair cache using the `yujianbo` metric with `min_sim=0.1`. The relaxed rows labeled `Same-label relaxed groups` do not use edit distance or semantic embeddings; they group explicit spans by character-offset overlap.

For the relaxed grouping, a candidate link is created between one span from each annotator when the spans have the same label and their character overlap covers at least 50% of the shorter span. The links form a bipartite graph, and connected components become disjoint span groups. This allows `1:N`, `M:1`, and `M:N` mappings without allowing a span node to appear in more than one group.

After grouping, direct explicit edges are collapsed to group-level edges. If an edge connects two spans that collapse into the same group, it is counted as an internal edge and excluded from group-to-group direct-edge agreement. The primary agreement score uses the same `edge-union` idea as the existing direct-edge table: a context is evaluated when at least one annotator has a positive edge. The all-pairs sensitivity also evaluates all ordered group pairs, adding many no/no contexts.

## Edge Agreement Summary

| Pairing strategy | Span pairing method | Contexts | Yes/Yes | A-only | B-only | Observed | Expected | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1:1 label-restricted edit | Edit distance (`yujianbo`, `min_sim=0.1`) | 95 | 7 | 64 | 24 | .074 | .414 | -.581 |
| Same-label relaxed groups | Offset overlap (`same-label`, threshold .5) | 51 | 11 | 31 | 9 | .216 | .430 | -.377 |

The relaxed grouping improves the positive-edge comparison: shared direct edges increase from 7 to 11, while A-only and B-only disagreements decrease substantially. However, kappa remains negative because the remaining edge decisions are still highly asymmetric.

## All-Pairs Sensitivity

| Pairing strategy | Span pairing method | Contexts | Yes/Yes | A-only | B-only | No/No | Observed | Expected | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1:1 label-restricted edit | Edit distance (`yujianbo`, `min_sim=0.1`) | 1,054 | 7 | 64 | 24 | 959 | .917 | .907 | .100 |
| Same-label relaxed groups | Offset overlap (`same-label`, threshold .5) | 576 | 11 | 31 | 9 | 525 | .931 | .897 | .323 |

The all-pairs version becomes positive because it includes many no/no contexts. This is useful as a sensitivity check, but it should not replace the stricter edge-union metric when the goal is to evaluate agreement over asserted support relations.

## Relaxed Group Structure

| Group arity | Count |
|---|---:|
| 1:1 | 54 |
| 1:2 | 15 |
| 1:3 | 3 |
| 1:4 | 1 |
| 1:5 | 2 |
| 2:1 | 5 |

Most non-1:1 groups are `1:N`: one span from `gdeyesu@umaryland.edu` maps to multiple spans from `zaidal-huneidi@umaryland.edu`. This supports the segmentation-style mismatch hypothesis: Zaid often uses finer-grained explicit spans.

## Split/Merge Groups By Label

| Label | 1:1 | 1:2 | 1:3 | 1:4 | 1:5 | 2:1 |
|---|---:|---:|---:|---:|---:|---:|
| Analysis | 15 | 6 | 1 | 1 | 2 | 2 |
| Background Facts | 7 | 2 | 0 | 0 | 0 | 1 |
| Conclusion | 8 | 0 | 0 | 0 | 0 | 0 |
| Procedural History | 9 | 0 | 0 | 0 | 0 | 1 |
| Rule | 15 | 7 | 2 | 0 | 0 | 1 |

The split/merge pattern is concentrated in `Analysis` and `Rule`, which are also the classes where longer argumentative spans are most likely to be segmented differently.

## Span Coverage

| Annotator | Explicit spans | Grouped | Ungrouped |
|---|---:|---:|---:|
| `gdeyesu@umaryland.edu` | 117 | 85 | 32 |
| `zaidal-huneidi@umaryland.edu` | 168 | 112 | 56 |

The relaxed method covers more than the strict one-to-one pairing, but many explicit spans remain ungrouped, especially for Zaid. This means some edge disagreement is still caused by unmatched endpoints.

## Edge Loss And Implicit Nodes

| Annotator | All edges | Explicit-explicit | Touch implicit | Grouped explicit | Internal in group | Lost endpoint | Collapsed group edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gdeyesu@umaryland.edu` | 88 | 79 | 9 | 44 | 2 | 35 | 42 |
| `zaidal-huneidi@umaryland.edu` | 163 | 89 | 74 | 40 | 19 | 49 | 20 |

The largest remaining asymmetry is graph style rather than span matching alone. Zaid has many more total edges and many more edges touching implicit nodes. Zaid also has 19 internal edges after grouping, compared with only 2 for Gregory. This suggests that some of Zaid's direct explicit-edge structure occurs inside finer-grained spans that collapse into broader Gregory spans.

## Per-Document Edge-Union Results

| Ref ID | Contexts | Yes/Yes | A-only | B-only | Kappa | Group arities |
|---|---:|---:|---:|---:|---:|---|
| 4 | 6 | 0 | 5 | 1 | -.385 | `1:1=2, 1:2=2, 1:3=1, 1:5=2` |
| 5 | 8 | 0 | 5 | 3 | -.882 | `1:1=6, 1:2=1` |
| 6 | 4 | 0 | 4 | 0 | .000 | `1:1=7, 1:2=2` |
| 7 | 3 | 1 | 2 | 0 | .000 | `1:1=4, 2:1=2` |
| 8 | 3 | 0 | 2 | 1 | -.800 | `1:1=3, 1:2=3, 1:3=2, 1:4=1, 2:1=1` |
| 9 | 5 | 1 | 4 | 0 | .000 | `1:1=10` |
| 10 | 4 | 3 | 1 | 0 | .000 | `1:1=5, 1:2=3` |
| 11 | 3 | 1 | 1 | 1 | -.500 | `1:1=7, 1:2=1` |
| 12 | 6 | 0 | 5 | 1 | -.385 | `1:1=5, 1:2=2` |
| 13 | 9 | 5 | 2 | 2 | -.286 | `1:1=5, 1:2=1, 2:1=2` |

The best signs of improvement are in documents 10 and 13, where group-level pairing recovers several shared edges. Documents 4, 5, 8, and 12 still show no shared positive group-level edges, so segmentation relaxation is not enough for those cases.

## Interpretation

The results support a mixed diagnosis. Span segmentation mismatch is real and affects edge agreement: many long spans from one annotator correspond to multiple shorter same-label spans from the other annotator, especially in `Analysis` and `Rule`. Relaxing the pairing to same-label connected components improves observed agreement from .074 to .216 and improves kappa from -.581 to -.377.

However, the agreement remains poor. The remaining disagreement appears to come mainly from graph-structure differences: different use of implicit intermediate nodes, different edge density, and edges that become internal after grouping. A natural next diagnostic would be path-relaxed edge agreement, where two grouped explicit spans count as connected if they are linked either directly or through implicit intermediate nodes.
