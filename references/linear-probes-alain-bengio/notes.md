# Linear Classifier Probes (Alain & Bengio 2016)


**Citation:** Guillaume Alain and Yoshua Bengio. "Understanding Intermediate Layers Using
Linear Classifier Probes." ICLR 2017 Workshop Track (submitted 2016).
arXiv:1610.01644; OpenReview: HJ4-rAVtl.

- arXiv abstract: https://arxiv.org/abs/1610.01644
- OpenReview: https://openreview.net/forum?id=HJ4-rAVtl

**Key idea.** Train small linear classifiers ("probes") entirely independently of the
frozen model on top of each layer's features, to monitor how suitable intermediate
representations are for classification. Applied to Inception v3 and ResNet-50, they
observe experimentally that linear separability of features increases monotonically with
depth ([arXiv abstract](https://arxiv.org/abs/1610.01644)). This is the foundational paper
of the whole probing paradigm: the probe is a diagnostic instrument external to the model,
and probe accuracy is read as "linearly accessible information" in a layer.

**Methodological lesson.** A probe must be (a) trained independently of the base model
(never backprop into the encoder), and (b) simple enough that its accuracy reflects the
representation rather than the probe's own learning capacity — the original choice is a
linear classifier. For time-series JEPA embeddings this is the baseline experiment: freeze
the encoder, fit linear/ridge probes for target properties (e.g., trend, periodicity,
anomaly labels) at different layers/heads, and compare against a random-feature baseline.
