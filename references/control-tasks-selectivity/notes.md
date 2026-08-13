# Control Tasks and Probe Selectivity (Hewitt & Liang 2019)


**Citation:** John Hewitt and Percy Liang. "Designing and Interpreting Probes with Control
Tasks." EMNLP-IJCNLP 2019, pp. 2733–2743.
arXiv:1909.03368; ACL Anthology: D19-1275.

- arXiv abstract: https://arxiv.org/abs/1909.03368
- ACL Anthology: https://aclanthology.org/D19-1275
- Author PDF: https://nlp.stanford.edu/pubs/hewitt2019control.pdf
- PDF in this folder: `hewitt-liang-2019-control-tasks.pdf`

**Key idea.** High probe accuracy is ambiguous: did the representation encode the property,
or did the probe just learn the task? **Control tasks** associate each word *type* with a
random output (same input structure, random labels), so by construction only the probe can
learn them. A faithful probe should be **selective**:
`selectivity = linguistic task accuracy − control task accuracy`
(definition verified from the PDF, Fig. 2 caption).
Findings ([arXiv abstract](https://arxiv.org/abs/1909.03368), PDF §4–5):

- Popular probes (incl. MLPs) on ELMo achieve high control-task accuracy — they are
  *not selective*: much of their "linguistic" accuracy is probe memorization capacity.
- Dropout, commonly used to limit probe complexity, is ineffective at improving MLP
  selectivity; other regularizers help.
- Layer conclusions can flip: ELMo layer 1 gives slightly better POS accuracy than layer 2,
  but layer-2 probes are substantially more selective — so "which layer encodes POS?" has
  no answer without a selectivity criterion.

**Methodological lesson.** Always pair every real probing task with a control task and
report selectivity, not accuracy alone. For time series: a control task assigns each
window *identity* a random label (conditioned on anything the probe could memorize); a
high-capacity probe on JEPA embeddings will score well on it, inflating naive conclusions.
Prefer constrained probes (linear or small, well-regularized) whose capacity is below the
task's intrinsic difficulty.
