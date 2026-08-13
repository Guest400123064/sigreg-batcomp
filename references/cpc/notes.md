# CPC — Contrastive Predictive Coding (applied to time series)

- **Citation:** Aäron van den Oord, Yazhe Li, Oriol Vinyals. arXiv:1807.03748, 2018
  (v2 Jan 2019).
- **URL:** <https://arxiv.org/abs/1807.03748>
- **Key idea:** Learn representations by *predicting the future in latent space* with
  powerful autoregressive models, trained with a probabilistic contrastive loss
  (InfoNCE) made tractable by negative sampling. The objective "induces the latent
  space to capture information that is maximally useful to predict future samples."
  (Verified from the arXiv abstract.) The paper demonstrates one architecture across
  four domains — speech, images, text, and RL in 3D environments — i.e., its
  time-series application is primarily speech/audio sequences.
- **What the embeddings capture (per the paper):** Information maximally predictive of
  future observations; the paper's framing is that the contrastive bound keeps
  slow-varying, predictive structure while discarding noise (the "InfoMax" view — this
  terminology is from the paper itself, see PDF). CPC is the conceptual ancestor of
  nearly all later predictive/contrastive TS methods and a standard baseline in TS
  representation papers (e.g., T-Loss, TNC compare against CPC-style objectives).
