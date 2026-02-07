"""
TAC'18 Online Physical Watermarking (Liu et al.) - Python implementation.

This package implements the *online* algorithm from the TAC paper (2018) for
designing a Gaussian watermark covariance and computing the Neyman–Pearson
detection statistic for replay attacks.

The implementation is based on:
  - the TAC paper (Algorithm 1 / Section IV), and
  - the authors' Julia reference implementation.

Primary entry point: `baseline.online.TACOnlineWatermarker`.
"""
