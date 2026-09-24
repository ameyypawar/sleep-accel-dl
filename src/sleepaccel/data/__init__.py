"""Data ingest: raw PhysioNet text files to a cached epoch tensor.

Pipeline order, and why it is this order:

1. :mod:`sleepaccel.data.labels` defines the epoch axis' vocabulary. The PSG
   label file is the authority on what an epoch *is*; everything else is
   resampled onto the grid it defines.
2. :mod:`sleepaccel.data.epochs` turns irregularly-sampled accelerometer
   samples into a fixed-size tensor per epoch, and decides which epochs are
   trustworthy enough to keep.
3. :mod:`sleepaccel.data.raw_io` parses the on-disk text files. It is written
   last, against the output of ``scripts/probe_raw_format.py``, because the
   delimiters and column order are not documented and must be measured.
"""
