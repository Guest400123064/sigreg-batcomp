# Amnesic Probing (Elazar et al. 2021)


**Citation:** Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg.
"Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals."
TACL 9:160–175, 2021. arXiv:2006.00995; DOI: 10.1162/tacl_a_00359.

- arXiv abstract: https://arxiv.org/abs/2006.00995
- TACL page: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00359/98090
- PDF in this folder: `elazar-2021-amnesic-probing.pdf`

**Key idea.** Standard probing measures what is *encoded*; it cannot support behavioral
claims about what the model *uses*. Amnesic probing instead builds a **counterfactual
representation** with a property removed and measures how downstream behavior changes —
"the utility of a property for a given task can be assessed by measuring the influence of a
causal intervention that removes it from the representation"
([arXiv abstract](https://arxiv.org/abs/2006.00995)).

**Methodology (detail, from the PDF §2.2).** Removal uses **Iterative Nullspace Projection
(INLP)** (Ravfogel et al. 2020): given labeled data for property Z, INLP trains linear
classifiers and projects the representations onto the intersection of their nullspaces,
iterated until no linear probe can predict Z above majority accuracy on a dev set; each
iteration strictly reduces the data matrix rank. The intervened ("amnesic")
representations are then passed through the frozen model to measure behavioral change
(e.g., masked-LM word prediction). Controls: remove the *same number of random directions*
to account for generic damage from rank reduction. Key empirical result on BERT:
**conventional probing performance is not correlated with task importance** — properties
that are highly probeable may be behaviorally irrelevant, and vice versa.

**Methodological lesson.** Separate two questions and answer both: (i) *encoding* — is
property X linearly present in the embedding (probes, MDL); (ii) *usage* — does removing X
change the model's behavior (amnesic/INLP-style intervention with random-direction
controls). For LeJEPA time-series embeddings, INLP removal of e.g. periodicity or level
information, followed by measuring change in the predictor's forecasting error, tests what
the predictive objective actually relies on — claims a probe alone cannot make. Caveat
(INLP removes only linearly-decodable directions, and removal can have side effects —
hence the random-direction control).
