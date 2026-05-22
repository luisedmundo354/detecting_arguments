Summary Of Weaknesses:
Missing concurrent and recent prior work — the novelty framing is overstated. The submission positions itself as filling a gap
in U.S. legal argument mining, but several closely related resources from 2024–2026 are not cited:

SCOTUS-Law / Belfathi et al., 2026 introduces a U.S. Supreme Court rhetorical-role dataset at three granularity levels.
Bongard, Held & Habernal, 2022 — "The Legal Argument Reasoning Task in Civil Procedure" — a U.S. civil-procedure legal-argument resource from the same lab as the cited Habernal et al. 2024 paper.
Chen et al., 2026 — Chinese judicial-decision tree-/graph-style argumentation annotation guidelines.
CLERC (Hou et al., NAACL Findings 2025) is cited but only in passing; it is the most natural retrieval baseline for U.S. case-law retrieval and should be discussed in §4.2.
The paper's claim of being "a novel resource for studying legal argument structure within … U.S. case law" needs to be reframed as "the first expert-annotated, tree-structured argument corpus for U.S. federal tax doctrine."
The IRAC framework itself is not novel. IRAC-based annotation/structuring is now mainstream in legal NLP — see LegalBench (Guha et al., 2023), LegalSemi (Kang et al., 2024), PILOT-Bench (Jang et al., 2026), PLAT (Choi et al., 2025), Servantez et
al., 2024 (Chain of Logic), and Yu et al., 2022 (Legal Prompting). The novelty lies in (a) the U.S. tax application and (b) the tree-structured edge annotations, not in the label inventory. The framing in §1 and §2 should be revised accordingly.

Edge / implicit-IC reliability is essentially zero (Table 3: κ = −0.59; Table 2c: κ ≈ −0.05). This is a fundamental concern,
because the tree-structure contribution is the principal methodological differentiator of this work over prior flat-span schemes. With observed agreement of 7% and per-annotator insertion rates of 14–28% (Table 2c), the released directed support graphs and
implicit-IC placeholders cannot be considered a reliable annotation layer in their current form. The Discussion notes the issue but does not propose a concrete mitigation; given that all retrieval queries (§4.2) are built from these unreliable trees, this casts doubt on what the structured-vs-flat retrieval comparison is actually measuring. At minimum, the authors should (i) report retrieval results restricted to subtrees where annotators agreed on edges, and (ii) attempt a second adjudication round on edges to estimate post-adjudication agreement.

Dataset is small (42 cases / 718 spans), and the double-annotated subset is even smaller (10 cases). Several label cells are
sparse: Conclusion has 44 explicit spans and Background Facts only 59. This drives the four-class merger but also limits any per-class conclusions, especially for the harder Rule/Analysis distinction. There is no statistical-significance testing for the classification
or retrieval comparisons (40 test queries), so differences such as FT-struct = 50% vs. BM25 = 47.5% hit@20 (Table 6, same-case-full) cannot be interpreted as robust effects.

The structural retrieval advantage is fragile. The fine-tuned structured retriever is best in only one of three regimes
(same-case full), and collapses (12.5% hit@20) under the global split while BM25 reaches 45%. Section §5 acknowledges this honestly, but the phrasing in the Abstract — "structured queries are most helpful for within-case argument completion" — risks being
read as a positive contribution when in fact the headline finding is closer to structure-aware fine-tuning fails to generalize across cases on a 4-case test set. This should be made more explicit in the Abstract and Conclusion.

Annotator pool and bias. Two law students (with one law-professor adjudicator) is a thin annotation base for a domain as
technical as Subchapter C reorganization doctrine. The paper does not characterize annotator background (tax-law coursework? prior practice?), training procedure, or annotation calibration rounds, all of which materially affect downstream reliability for legal
corpora (Habernal et al., 2024 report a substantially more elaborate calibration protocol).

Baselines are missing for the U.S. functional-segmentation precedent. Šavelka and Ashley
(2018) propose a 7-class functional segmentation for U.S. opinions, with overlapping categories (Background, Analysis, Conclusions). The paper would benefit from either (i) projecting the proposed labels onto that scheme or (ii) running their model/labels as a baseline, to contextualize the 5-/4-class results.

GPT-5-mini comparison is under-specified. The paper labels GPT-5-mini as a "context-rich upper bound" but the prompt template,
label-description text, context-construction protocol, decoding parameters, version date, and zero/few-shot setup are not described in the main body or, as far as I can find, the appendix. This makes the row uninterpretable and unreproducible. Either remove or document fully.

No human topline for retrieval. The argument-completion task does not include a human-completion or oracle baseline, so the
absolute retrieval numbers (hit@20 ≈ 50%) cannot be interpreted as easy/hard. A second-annotator topline on a small held-out subset would substantially strengthen §4.2.

Format and presentation issues. Figure 1 is dense and difficult to parse without zooming; the per-span text is partially
elided, which makes the syllogism-tree concept hard to grasp on first read. Figure 3 has three separate panels with separate y-axes — a single panel with shaded regions or a small-multiples layout with shared scaling would improve clarity. Table 4’s "Macro Avg"
formatting is inconsistent (italicized scores have spaces inside the decimal: "0 .75").

Limitations section is thin. The paper mentions corpus size and domain narrowness, but does not address (i) the negative edge κ as a limitation of the tree-structure contribution itself, (ii) potential annotator bias (only two students), or (iii) the fact that I.R.C. §368 doctrine has been amended over the dataset's time range (the cases span pre-1986, post-1986, and post-TCJA tax law). These materially affect external validity.

Comments Suggestions And Typos:
The Habernal et al. (2024) paper concerns the ECHR, not U.S. or German court decisions. The wording in §2 is correct — please
retain this clarity in revision.
Consider adding the Xu and Ashley, 2022 multi-granularity argument-mining work and the MARRO and LegalSeg corpora to §2 for a more complete landscape.
§3.3 ("Adjudication"): clarify whether adjudicator decisions are at the case or the annotation level. The text is slightly ambiguous: "Decisions are made for the entire case rather than for individual components" suggests the former, but later "Suggestions
and corrections to annotations are provided only when they conflict with the established guidelines" suggests targeted edits. A worked example would help.
§4.1: report standard deviations across the five folds for Macro-F1, not only point estimates.
§4.2: please report MRR for all settings (Appendix B has it, but it should be discussed in the main text). Also clarify whether E5 = intfloat/e5-base-v2 exactly (Appendix B implies so but main text says "E5").
§4.2 retrieval: report whether the FT-struct/FT-flat retrievers were tuned on a development set, and how (val set size = 28 is
small).
Add a "data statement" in the spirit of Bender & Friedman, 2018: annotators’ legal training, jurisdiction, time period of cases, statutory regime version, decoding/segmentation tool used, etc.
Consider releasing case IDs and citations in a machine-readable manifest so others can reuse with their own pre-processing.
Typo / spacing: Table 4 has many 0 .77 style numbers — the space between digit and decimal point should be removed in
camera-ready.
Figure 1 and Figure 4 caption text contains characters that look like they were lost in PDF extraction ([. . . ]). If this is the actual rendered text, consider replacing with an ellipsis character "…" for visual cleanliness.
Abstract: replace "novel resource for studying legal argument structure within … U.S. case law" with a more precise scope phrase such as "the first expert-annotated, tree-structured argument corpus for U.S. federal tax case law on corporate reorganizations under
I.R.C. §368."