"""
Agent modules for Hybrid-46 v2.
Each agent has a distinct architecture for diversity.
"""

from .agent_a import AgentA
from .agent_b import AgentB
from .agent_c import AgentC
from .agent_k import AgentK
from .agent_2d import Agent2D
from .agent_t import AgentT
from .agent_q import AgentQ
from .agent_tq import AgentTQ
from .agent_h import AgentH
from .agent_m import AgentM
from .agent_vix import AgentVIX

__all__ = [
    'AgentA', 'AgentB', 'AgentC',
    'AgentK', 'Agent2D',
    'AgentT', 'AgentQ', 'AgentTQ', 'AgentH', 'AgentM', 'AgentVIX',
]
