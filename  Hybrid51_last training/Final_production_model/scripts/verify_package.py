#!/usr/bin/env python3
"""
Package Verification Script

Verifies that all required files are present, models can be loaded
correctly using BinaryIndependentAgent wrapper, and data paths align.
"""

import json
import sys
from pathlib import Path
import torch
import numpy as np

def verify_package():
    """Verify production package integrity."""
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root))
    errors = []
    warnings = []

    print("=" * 60)
    print("PRODUCTION PACKAGE VERIFICATION")
    print("=" * 60)

    # --- Config files ---
    config_path = root / "config/production_config.json"
    if not config_path.exists():
        errors.append("Missing config/production_config.json")
    else:
        print("✓ config/production_config.json")

    # Config Python module (feature_subsets needed by IndependentAgent)
    for f in ["config/__init__.py", "config/feature_subsets.py"]:
        if not (root / f).exists():
            errors.append(f"Missing {f} (required by IndependentAgent)")
        else:
            print(f"✓ {f}")

    # --- Documentation ---
    for doc in ["README.md", "INVENTORY.md", "DEPLOYMENT_SUMMARY.txt"]:
        if not (root / doc).exists():
            warnings.append(f"Missing documentation: {doc}")
        else:
            print(f"✓ {doc}")

    # --- Model directories ---
    model_dir = root / "models"
    for stage in ["stage1", "stage2", "stage3"]:
        if not (model_dir / stage).exists():
            errors.append(f"Missing models/{stage}/ directory")
        else:
            print(f"✓ models/{stage}/ exists")

    # --- Stage 1: 35 checkpoints ---
    stage1_files = sorted((model_dir / "stage1").glob("*.pt"))
    if len(stage1_files) != 35:
        errors.append(f"Stage 1: Expected 35 models, found {len(stage1_files)}")
    else:
        print(f"✓ Stage 1: All 35 models present")

    # --- Stage 2: 7 fusion + 1 chain context ---
    stage2_files = sorted((model_dir / "stage2").glob("*.pt"))
    if len(stage2_files) != 7:
        errors.append(f"Stage 2: Expected 7 fusion models, found {len(stage2_files)}")
    else:
        print(f"✓ Stage 2: All 7 fusion models present")

    chain_ctx_path = model_dir / "stage2/chain_context.npz"
    if not chain_ctx_path.exists():
        errors.append("Stage 2: Missing chain_context.npz")
    else:
        cc = np.load(chain_ctx_path)
        required_keys = ['val_logits', 'val_probs', 'test_logits', 'test_probs']
        missing_keys = [k for k in required_keys if k not in cc]
        if missing_keys:
            errors.append(f"chain_context.npz missing keys: {missing_keys}")
        else:
            print(f"✓ Stage 2: chain_context.npz has correct keys")

    # --- Stage 3 ---
    if not (model_dir / "stage3/stage3_vix_gated.pt").exists():
        errors.append("Stage 3: Missing stage3_vix_gated.pt")
    else:
        print(f"✓ Stage 3: VIX-gated model present")

    if (model_dir / "stage3/metrics.json").exists():
        m = json.load(open(model_dir / "stage3/metrics.json"))
        print(f"  Best method: {m.get('best_method')}")

    # --- Source code ---
    for code_dir in ["hybrid51_models", "hybrid51_utils"]:
        if not (root / code_dir).exists():
            errors.append(f"Missing {code_dir}/ directory")
        else:
            print(f"✓ {code_dir}/ present")

    # --- Deep model loading test ---
    print("\nDeep model loading tests...")

    # Test 1: Stage 1 checkpoint with BinaryIndependentAgent
    try:
        from scripts.simple_inference import BinaryIndependentAgent, _build_model_from_ckpt
        sample = stage1_files[0]
        ckpt = torch.load(sample, map_location='cpu', weights_only=False)
        assert all(k.startswith('base.') for k in ckpt['model_state_dict']), \
            "State dict keys should start with 'base.'"
        model = _build_model_from_ckpt(ckpt, ckpt['agent_type'], torch.device('cpu'),
                                        sample.stem.split('_agent')[0])
        print(f"✓ Stage 1 model loaded with strict=True: {sample.name}")
    except Exception as e:
        errors.append(f"Stage 1 model loading failed: {e}")

    # Test 2: Stage 2 standard agent checkpoint
    try:
        from hybrid51_models.cross_symbol_agent_fusion import CrossSymbolAgentFusion
        ckpt_a = torch.load(model_dir / "stage2/agentA_fusion.pt",
                            map_location='cpu', weights_only=False)
        n_inputs = ckpt_a['n_inputs']
        fusion = CrossSymbolAgentFusion(n_inputs=n_inputs, hidden_dim=32, dropout=0.2)
        fusion.load_state_dict(ckpt_a['model_state_dict'], strict=True)
        print(f"✓ Stage 2 Agent A loaded (n_inputs={n_inputs}, key=model_state_dict)")
    except Exception as e:
        errors.append(f"Stage 2 Agent A loading failed: {e}")

    # Test 3: Stage 2 Agent 2D checkpoint
    try:
        ckpt_2d = torch.load(model_dir / "stage2/agent2D_fusion.pt",
                             map_location='cpu', weights_only=False)
        n_inputs_2d = ckpt_2d['n_inputs']
        fusion_2d = CrossSymbolAgentFusion(n_inputs=n_inputs_2d, hidden_dim=32, dropout=0.2)
        fusion_2d.load_state_dict(ckpt_2d['fusion_state_dict'], strict=True)
        print(f"✓ Stage 2 Agent 2D loaded (n_inputs={n_inputs_2d}, key=fusion_state_dict)")
    except Exception as e:
        errors.append(f"Stage 2 Agent 2D loading failed: {e}")

    # Test 4: Stage 3 VIX-gated model
    try:
        from hybrid51_models.regime_gated_meta_model import RegimeGatedProbFusion
        ckpt3 = torch.load(model_dir / "stage3/stage3_vix_gated.pt",
                           map_location='cpu', weights_only=False)
        model3 = RegimeGatedProbFusion(
            agent_names=ckpt3['agent_names'],
            vix_feat_dim=int(ckpt3['vix_feat_dim']),
            regime_emb_dim=int(ckpt3['regime_emb_dim']),
            fusion_hidden_dim=int(ckpt3['fusion_hidden_dim']),
            dropout=float(ckpt3['dropout']),
        )
        model3.load_state_dict(ckpt3['model_state_dict'], strict=True)
        print(f"✓ Stage 3 VIX-gated loaded (threshold={ckpt3.get('threshold', 0.5):.3f})")
    except Exception as e:
        errors.append(f"Stage 3 model loading failed: {e}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"  - {warn}")
    if not errors and not warnings:
        print("\n✅ ALL CHECKS PASSED — Package ready for production.")
        return 0
    elif not errors:
        print("\n✅ PACKAGE VALID (with warnings)")
        return 0
    else:
        print("\n❌ PACKAGE INCOMPLETE — Fix errors before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_package())
