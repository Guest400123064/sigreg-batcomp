# Structural Probes (Hewitt & Manning 2019)


**Citation:** John Hewitt and Christopher D. Manning. "A Structural Probe for Finding
Syntax in Word Representations." NAACL-HLT 2019, pp. 4129–4138.
ACL Anthology: N19-1419.

- ACL Anthology page: https://aclanthology.org/N19-1419/
- Author PDF: https://nlp.stanford.edu/pubs/hewitt2019structural.pdf
- No arXiv version located; ACL Anthology is the canonical source (verified abstract there).

**Key idea.** Probe for *structure*, not just scalar properties: does the embedding space
contain a linear transformation under which geometry mirrors a parse tree? The probe learns
a linear map B such that squared L2 distance between transformed word vectors encodes
tree distance between the words, and squared L2 norm encodes depth in the parse tree.
Such transformations exist for ELMo and BERT but not for non-contextual baselines —
evidence that entire syntax trees are embedded implicitly in the vector geometry of deep
models ([ACL abstract](https://aclanthology.org/N19-1419/)).

**Methodology (detail).** The probe is a single learned linear projection optimized to make
`||B(h_i - h_j)||^2` approximate gold tree distance d_T(w_i, w_j) (and `||B h_i||^2`
approximate parse depth). The hypothesis class is deliberately restricted to linear maps so
that success means the structure is *linearly present* in the representation geometry.

**Methodological lesson.** Probing questions can target relational/structural hypotheses,
not just label prediction. The time-series analogue: learn a linear map under which
embedding distance between windows encodes lag distance, phase offset, or hierarchical
temporal structure — a much sharper claim about "what the embedding knows" than
classification accuracy on hand-picked labels.
