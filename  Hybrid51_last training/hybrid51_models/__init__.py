"""
Hybrid51 Models - v2 (Stage 1-3 Refactored)

Changes from v1:
- All BatchNorm replaced with LayerNorm/GroupNorm for stability
- Feature subsetting support in IndependentAgent
- Input normalization in Agent T and Q
- Residual connections in Agent A
"""

from .backbone import TemporalBackbone, TemporalBackboneWithAttention
from .independent_agent import IndependentAgent, create_independent_agent
from .regime_gated_meta_model import RegimeGatedProbFusion
from .tlt_gated_agent_fusion import TLTGatedAgentFusion
