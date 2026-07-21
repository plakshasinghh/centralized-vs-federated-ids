"""Live network traffic capture and flow-level feature extraction.

Distinct from `src.data`, which loads and preprocesses an already-collected
CSV dataset. This package sniffs live packets (via scapy) and extracts the
same kind of flow-level features (packet counts, byte counts, TCP flags,
entropy, etc.) in real time, so it can feed a trained model for live
inference or be used to build new labeled data.
"""
