# RankMe (Garrido et al. 2023)


**Citation:** Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann LeCun.
"RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations
by Their Rank." ICML 2023, PMLR 202:10929–10974. arXiv:2210.02885.

- PMLR page (with abstract and bibtex): https://proceedings.mlr.press/v202/garrido23a.html
- arXiv abstract: https://arxiv.org/abs/2210.02885

**Key idea.** Joint-embedding SSL (JE-SSL) has no reconstruction and uninformative loss
values, so on a label-free domain there is no signal of training success. RankMe shows the
**effective rank of the embedding matrix** (entropy of the singular values, exponentiated)
is a simple, training-free, hyperparameter-free, label-free criterion indicative of
downstream performance — even across downstream datasets — and enables label-free
hyperparameter selection with nearly no loss vs. label-based selection, validated over
hundreds of training runs ([PMLR abstract](https://proceedings.mlr.press/v202/garrido23a.html)).

**Methodological lesson.** Directly targeted at JE architectures (the same family as
JEPA/LeJEPA): before probing *content*, quantify the embedding's **dimensionality usage**.
Effective rank on a held-out batch of time-series windows is a cheap sanity metric that
detects collapse and predicts downstream linear-probe performance without any labels —
the right first diagnostic for a new LeJEPA time-series training run.
