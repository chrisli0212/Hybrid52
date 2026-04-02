from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

# Production training / inference use the 30-minute label horizon (Tier3: .../horizon_30min/).
# Older experiments sometimes used 15; CLI always accepts --horizon explicitly.
DEFAULT_TRAINING_HORIZON_MINUTES = 30


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    data_root: Path

    @staticmethod
    def default() -> "ArtifactPaths":
        # Env vars:
        #   HYBRID55_ARTIFACT_ROOT  — override model checkpoint/results root
        #   HYBRID55_DATA_ROOT      — override tier3 binary data root
        #                             default: /workspace/data/tier3_binary_hybrid55
        # Backward compatibility: HYBRID52_* env vars are still honored as fallbacks.
        root = Path(
            os.getenv(
                "HYBRID55_ARTIFACT_ROOT",
                os.getenv("HYBRID52_ARTIFACT_ROOT", str(Path(__file__).resolve().parent.parent)),
            )
        )
        data_root = Path(
            os.getenv(
                "HYBRID55_DATA_ROOT",
                os.getenv("HYBRID52_DATA_ROOT", "/workspace/data/tier3_binary_hybrid55"),
            )
        )
        return ArtifactPaths(root=root, data_root=data_root)

    @property
    def stage1_results(self) -> Path:
        return self.root / "results" / "stage1"

    @property
    def stage2_results(self) -> Path:
        return self.root / "results" / "stage2"

    @property
    def stage3_results(self) -> Path:
        return self.root / "results" / "stage3"

    def tier3_dir(self, symbol: str, horizon: int) -> Path:
        return self.data_root / symbol / f"horizon_{horizon}min"

    def stage1_ckpt(self, symbol: str, agent: str, horizon: int) -> Path:
        return self.stage1_results / f"{symbol}_agent{agent}_classifier_h{horizon}.pt"

    def stage2_ckpt(self, target: str, pair: str, horizon: int) -> Path:
        return self.stage2_results / f"{target}_{pair}_h{horizon}_pair_fusion.pt"

    def stage2_probs(self, target: str, pair: str, horizon: int) -> Path:
        return self.stage2_results / f"{target}_{pair}_h{horizon}_pair_probs.npz"

    def stage3_metrics(self, target: str, horizon: int) -> Path:
        return self.stage3_results / f"{target}_h{horizon}_stage3_meta_metrics.json"

    def stage3_logreg_model(self, target: str, horizon: int) -> Path:
        return self.stage3_results / f"{target}_h{horizon}_stage3_meta.joblib"

    def stage3_mlp_model(self, target: str, horizon: int) -> Path:
        return self.stage3_results / f"{target}_h{horizon}_stage3_meta_mlp.pt"

    def stage3_cross_agent_vix_model(self, target: str, horizon: int) -> Path:
        return self.stage3_results / f"{target}_h{horizon}_stage3_cross_agent_vix_gated.pt"

    @property
    def stage2_cross_results(self) -> Path:
        return self.root / "results" / "stage2_cross"

    def stage2_chain_context(self, symbol: str, horizon: int) -> Path:
        return self.stage2_cross_results / f"{symbol}_h{horizon}_chain_context.npz"

    def stage2_per_agent_ckpt(self, target: str, agent: str, horizon: int) -> Path:
        return self.stage2_cross_results / f"{target}_agent{agent}_h{horizon}_cross_fusion.pt"

    def stage2_per_agent_probs(self, target: str, agent: str, horizon: int) -> Path:
        return self.stage2_cross_results / f"{target}_agent{agent}_h{horizon}_cross_probs.npz"

    def stage2_tlt_gated_ckpt(self, target: str, agent: str, horizon: int) -> Path:
        return self.stage2_cross_results / f"{target}_agent{agent}_h{horizon}_tlt_gated_fusion.pt"

    def stage2_tlt_gated_probs(self, target: str, agent: str, horizon: int) -> Path:
        return self.stage2_cross_results / f"{target}_agent{agent}_h{horizon}_tlt_gated_probs.npz"

    def stage1_2d_ckpt(self, symbol: str, horizon: int) -> Path:
        chain_only = self.root / "results" / "stage1_2d_chain_only" / f"{symbol}_agent2D_classifier_h{horizon}.pt"
        if chain_only.exists():
            return chain_only
        return self.stage1_results / f"{symbol}_agent2D_classifier_h{horizon}.pt"

    def tier3_chain_dir(self, symbol: str, horizon: int) -> Path:
        return self.tier3_dir(symbol, horizon)

    @property
    def stage1_vix_results(self) -> Path:
        return self.root / "results" / "stage1_vix"

    def stage1_vix_ckpt(self) -> Path:
        return self.stage1_vix_results / "vix_agent_best.pt"

    @property
    def tier3_vix_root(self) -> Path:
        return Path("/workspace/data/tier3_vix_hybrid55")

    def tier3_vix_dir(self, symbol: str = "VIXW") -> Path:
        return self.tier3_vix_root / symbol
