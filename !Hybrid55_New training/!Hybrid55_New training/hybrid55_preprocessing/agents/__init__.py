"""
Per-agent assemblers.
Each agent sub-package has:
  - feature_config.py  — AGENT_X_DIM, feature name list, group definitions
  - extractor.py       — AgentXExtractor: selects blocks from shared extractors,
                          assembles vector, asserts shape, runs zero alerts
  - validator.py       — standalone validation / unit-test helpers
"""
