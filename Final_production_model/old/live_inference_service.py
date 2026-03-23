from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from live_model_bridge import LiveFeatureBridge, REQUIRED_SYMBOLS


MODEL_ROOT = Path("/workspace/Final_production_model")
MODEL_DIR = MODEL_ROOT / "models"
CONFIG_PATH = MODEL_ROOT / "config/production_config.json"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from hybrid51_models.independent_agent import IndependentAgent  # noqa: E402
from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion  # noqa: E402
from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion  # noqa: E402


ALL_AGENTS = ["A", "B", "C", "K", "T", "Q", "2D"]


@dataclass
class _Stage1Bundle:
    model: nn.Module
    norm_mean: np.ndarray | None
    norm_std: np.ndarray | None


class BinaryIndependentAgent(nn.Module):
    """Wrapper matching checkpoint key format from production training."""

    def __init__(
        self,
        agent_type: str,
        feat_dim: int = 325,
        temporal_dim: int = 128,
        dropout: float = 0.2,
        use_feature_subset: bool = True,
        use_attention_backbone: bool = False,
        use_attention_pool: bool = False,
        cls_input_dim: int | None = None,
    ):
        super().__init__()
        self.base = IndependentAgent(
            agent_type=agent_type,
            feat_dim=feat_dim,
            temporal_dim=temporal_dim,
            dropout=dropout,
            num_classes=5,
            use_feature_subset=use_feature_subset,
            use_attention_backbone=use_attention_backbone,
            use_attention_pool=use_attention_pool,
        )
        if cls_input_dim is None:
            cls_input_dim = (2 + temporal_dim) if self.base.use_backbone else (2 + 32)
        self.base.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, sequences: torch.Tensor, chain_2d: torch.Tensor | None = None) -> torch.Tensor:
        return self.base(sequences, chain_2d=chain_2d).squeeze(-1)


def _build_model_from_ckpt(ckpt: dict, agent_type: str, device: torch.device, symbol: str) -> nn.Module:
    state = ckpt["model_state_dict"]
    if "feat_dim" in ckpt:
        feat_dim = int(ckpt["feat_dim"])
    elif "base._feat_idx" in state and state["base._feat_idx"].numel() > 0:
        feat_dim = 650 if int(state["base._feat_idx"].max().item()) >= 325 else 325
    else:
        feat_dim = 325 if symbol == "SPXW" else 650

    use_subset = bool(ckpt.get("feature_subset", True))
    use_attn_bb = bool(ckpt.get("use_attention_backbone", False))
    use_attn_pool = bool(ckpt.get("use_attention_pool", False))
    cls_in_dim = int(state["base.classifier.0.weight"].shape[1])
    has_static_proj = "base.static_proj.weight" in state

    model = BinaryIndependentAgent(
        agent_type=agent_type,
        feat_dim=feat_dim,
        use_feature_subset=use_subset,
        use_attention_backbone=use_attn_bb,
        use_attention_pool=use_attn_pool,
        cls_input_dim=cls_in_dim,
    ).to(device)

    if not has_static_proj and hasattr(model.base, "static_proj"):
        del model.base.static_proj

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class LiveHybrid51InferenceService:
    """
    Persistent live inference service.

    Loads production checkpoints once, builds real-time features via bridge,
    and returns a single latest prediction for the dashboard.
    """

    def __init__(self, model_root: Path = MODEL_ROOT, device: str | None = None):
        self.model_root = Path(model_root)
        self.model_dir = self.model_root / "models"
        self.config = json.loads(CONFIG_PATH.read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bridge = LiveFeatureBridge()

        self.stage1: Dict[str, Dict[str, _Stage1Bundle]] = {s: {} for s in REQUIRED_SYMBOLS}
        self.stage2: Dict[str, Tuple[nn.Module, dict]] = {}
        self.stage3_model: RegimeGatedProbFusion | None = None
        self.stage3_threshold: float = 0.5
        self.stage3_agent_order = list(ALL_AGENTS)

        self._load_all_models()

    def _load_norm_stats(self, symbol: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
        horizon = int(self.config.get("model_info", {}).get("horizon_minutes", 30))
        data_root = Path(self.config.get("data_paths", {}).get("tier3_binary_root", ""))
        d = data_root / symbol / f"horizon_{horizon}min"
        nm_path = d / "norm_mean.npy"
        ns_path = d / "norm_std.npy"
        if nm_path.exists() and ns_path.exists():
            try:
                return np.load(nm_path), np.load(ns_path)
            except Exception:
                return None, None
        return None, None

    def _load_all_models(self) -> None:
        for symbol in REQUIRED_SYMBOLS:
            nm, ns = self._load_norm_stats(symbol)
            for agent in ALL_AGENTS:
                ckpt_path = self.model_dir / f"stage1/{symbol}_agent{agent}.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model = _build_model_from_ckpt(ckpt, agent_type=agent, device=self.device, symbol=symbol)
                self.stage1[symbol][agent] = _Stage1Bundle(model=model, norm_mean=nm, norm_std=ns)

        for agent in ALL_AGENTS:
            ckpt_path = self.model_dir / f"stage2/agent{agent}_fusion.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            n_inputs = int(ckpt["n_inputs"])
            fusion = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2).to(self.device)
            sd_key = "fusion_state_dict" if agent == "2D" else "model_state_dict"
            fusion.load_state_dict(ckpt[sd_key], strict=True)
            fusion.eval()
            self.stage2[agent] = (fusion, ckpt)

        ckpt3 = torch.load(self.model_dir / "stage3/stage3_vix_gated.pt", map_location="cpu", weights_only=False)
        self.stage3_model = RegimeGatedProbFusion(
            agent_names=ckpt3["agent_names"],
            vix_feat_dim=int(ckpt3["vix_feat_dim"]),
            regime_emb_dim=int(ckpt3["regime_emb_dim"]),
            fusion_hidden_dim=int(ckpt3["fusion_hidden_dim"]),
            dropout=float(ckpt3["dropout"]),
        ).to(self.device)
        self.stage3_model.load_state_dict(ckpt3["model_state_dict"], strict=True)
        self.stage3_model.eval()
        self.stage3_threshold = float(ckpt3.get("threshold", 0.5))
        self.stage3_agent_order = list(ckpt3["agent_names"])

    @torch.no_grad()
    def _stage1_predict(self, seq: np.ndarray, chain: np.ndarray | None, bundle: _Stage1Bundle) -> Tuple[float, float]:
        x = torch.from_numpy(seq.astype(np.float32)).to(self.device)
        if bundle.norm_mean is not None and bundle.norm_std is not None:
            nm = torch.from_numpy(bundle.norm_mean.astype(np.float32)).to(self.device)
            ns = torch.from_numpy(bundle.norm_std.astype(np.float32)).to(self.device)
            x = (x - nm) / torch.clamp(ns, min=1e-6)
        c = None
        if chain is not None:
            c = torch.from_numpy(chain.astype(np.float32)).to(self.device)
        logits = bundle.model(x, chain_2d=c).detach().cpu().numpy().reshape(-1)
        logit = float(logits[-1])
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        return logit, prob

    @torch.no_grad()
    def predict_latest(self, agg_df, snap_df) -> Dict[str, Any]:
        t0 = time.perf_counter()
        bridge_out = self.bridge.build_from_dataframes(agg_df, snap_df)
        diag = dict(bridge_out.diagnostics)

        if diag.get("suppression_reason"):
            return {
                "ok": False,
                "suppressed": True,
                "reason": diag["suppression_reason"],
                "prob": 0.5,
                "pred": 0,
                "threshold": self.stage3_threshold,
                "gates": {a: 0.5 for a in ALL_AGENTS},
                "stage2_probs": {a: 0.5 for a in ALL_AGENTS},
                "confidence": 0.0,
                "signal_strength": 0.0,
                "diagnostics": {**diag, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)},
            }

        stage1_logits: Dict[str, Dict[str, float]] = {s: {} for s in REQUIRED_SYMBOLS}
        stage1_probs: Dict[str, Dict[str, float]] = {s: {} for s in REQUIRED_SYMBOLS}
        missing_stage1 = []

        for symbol in REQUIRED_SYMBOLS:
            seq = bridge_out.stage1_sequences[symbol]
            chain = bridge_out.stage1_chain_2d[symbol]
            for agent in ALL_AGENTS:
                bundle = self.stage1.get(symbol, {}).get(agent)
                if bundle is None:
                    missing_stage1.append(f"{symbol}:{agent}")
                    continue
                lg, pb = self._stage1_predict(seq, chain if agent == "2D" else None, bundle)
                stage1_logits[symbol][agent] = lg
                stage1_probs[symbol][agent] = pb

        stage2_probs: Dict[str, float] = {}
        for agent in ALL_AGENTS:
            if agent not in self.stage2:
                continue
            fusion, ckpt = self.stage2[agent]
            if agent == "2D":
                active_peers = ckpt.get("active_peers", ["SPY", "QQQ", "IWM", "TLT"])
                spxw_logit = stage1_logits["SPXW"].get("2D", 0.0)
                spxw_prob = stage1_probs["SPXW"].get("2D", 0.5)
                parts = [spxw_logit, spxw_prob]
                peer_logits_only = []
                for sym in active_peers:
                    peer_l = stage1_logits.get(sym, {}).get("2D", 0.0)
                    peer_p = stage1_probs.get(sym, {}).get("2D", 0.5)
                    parts.extend([peer_l, peer_p])
                    peer_logits_only.append(peer_l)
                for peer_l in peer_logits_only:
                    parts.append(spxw_logit - peer_l)
                X = np.asarray(parts, dtype=np.float32).reshape(1, -1)
            else:
                syms = ckpt.get("symbols", ["SPXW", "SPY", "QQQ", "IWM"])
                peer_syms = [s for s in syms if s != "SPXW"]
                spxw_logit = stage1_logits["SPXW"].get(agent, 0.0)
                parts = [spxw_logit, stage1_probs["SPXW"].get(agent, 0.5)]
                for sym in peer_syms:
                    parts.append(stage1_logits.get(sym, {}).get(agent, 0.0))
                    parts.append(stage1_probs.get(sym, {}).get(agent, 0.5))
                for sym in peer_syms:
                    parts.append(spxw_logit - stage1_logits.get(sym, {}).get(agent, 0.0))
                # Live proxy for chain context in stage2 standard agents.
                ctx_logit = stage1_logits["SPXW"].get("2D", 0.0)
                ctx_prob = stage1_probs["SPXW"].get("2D", 0.5)
                parts.extend([ctx_logit, ctx_prob])
                X = np.asarray(parts, dtype=np.float32).reshape(1, -1)

            n_inputs = int(ckpt["n_inputs"])
            if X.shape[1] != n_inputs:
                if X.shape[1] < n_inputs:
                    pad = np.zeros((1, n_inputs - X.shape[1]), dtype=np.float32)
                    X = np.concatenate([X, pad], axis=1)
                else:
                    X = X[:, :n_inputs]

            logits = fusion(torch.from_numpy(X).to(self.device)).detach().cpu().numpy().reshape(-1)
            prob = float(1.0 / (1.0 + np.exp(-float(logits[-1]))))
            stage2_probs[agent] = prob

        if self.stage3_model is None:
            raise RuntimeError("Stage3 model not loaded")

        agent_cols = []
        for a in self.stage3_agent_order:
            agent_cols.append([stage2_probs.get(a, 0.5)])
        agent_mat = np.asarray(agent_cols, dtype=np.float32).T
        vix_feat = bridge_out.vix_features.astype(np.float32)
        if vix_feat.shape[1] != 10:
            vix_feat = np.pad(vix_feat, ((0, 0), (0, max(0, 10 - vix_feat.shape[1]))), constant_values=0.0)[:, :10]

        logits3, gates3, _ = self.stage3_model(
            torch.from_numpy(agent_mat).to(self.device),
            torch.from_numpy(vix_feat).to(self.device),
        )
        prob = float(torch.sigmoid(logits3).detach().cpu().numpy().reshape(-1)[-1])
        pred = int(prob > self.stage3_threshold)
        gates = gates3.detach().cpu().numpy().reshape(-1)
        gates_map = {a: float(gates[i]) for i, a in enumerate(self.stage3_agent_order)}

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        confidence = float(min(1.0, abs(prob - self.stage3_threshold) * 2.0))
        signal_strength = float(np.clip((prob - self.stage3_threshold) * 2.0, -1.0, 1.0))
        return {
            "ok": True,
            "suppressed": False,
            "reason": None,
            "prob": prob,
            "pred": pred,
            "threshold": self.stage3_threshold,
            "gates": gates_map,
            "stage2_probs": {a: float(stage2_probs.get(a, 0.5)) for a in ALL_AGENTS},
            "confidence": confidence,
            "signal_strength": signal_strength,
            "diagnostics": {
                **diag,
                "stage1_missing_count": len(missing_stage1),
                "stage2_agents_ready": sorted(stage2_probs.keys()),
                "latency_ms": latency_ms,
            },
        }
