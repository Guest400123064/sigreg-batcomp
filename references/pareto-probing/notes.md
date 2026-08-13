# Pareto Probing (Pimentel et al. 2020)


**Citation:** Tiago Pimentel, Naomi Saphra, Adina Williams, and Ryan Cotterell.
"Pareto Probing: Trading Off Accuracy for Complexity." EMNLP 2020, pp. 3138–3153.
arXiv:2010.02180.

- arXiv abstract: https://arxiv.org/abs/2010.02180
- ACL Anthology: https://aclanthology.org/2020.emnlp-main.254

**Key idea.** Probing metrics should reflect the fundamental trade-off between probe
**complexity and performance**; the paper proposes the **Pareto hypervolume** (area
dominated by the accuracy–complexity Pareto frontier across a family of probes) as the
metric, with several parametric and non-parametric complexity measures. Two critical
findings ([arXiv abstract](https://arxiv.org/abs/2010.02180)):

- Under Pareto evaluation, results defy intuitions — e.g., non-contextual fastText
  representations encode *more* morpho-syntactic information than contextual BERT
  representations on simple tasks (POS tagging, dependency arc labeling).
- Simple probing tasks are inadequate to assess structure in contextual representations;
  the authors propose **full dependency parsing** as a probing task, where a wide gap
  between contextual and non-contextual representations appears.

**Relation to MDL probing.** This is a direct response to Voita & Titov 2020: it shares the
"price in probe effort" motivation but replaces codelength with an explicit
complexity–accuracy frontier. (A companion paper by Pimentel et al., "Information-Theoretic
Probing for Linguistic Structure", ACL 2020, pp. 4609–4622, develops a mutual-information
view of probing with control-task baselines — title/venue verified via reference lists in
[arXiv:2104.05904](https://arxiv.org/pdf/2104.05904); not independently fetched.)

**Methodological lesson.** Probe-task choice is part of the measurement instrument: tasks
that are too easy let weak/linear-access representations score as well as strong ones.
Design time-series probing tasks at multiple difficulties (linear trend/slope → regime
classification → full next-window forecasting), and report an accuracy–complexity
trade-off rather than a single probe's accuracy.
