"""The final200 chain configs map onto the current train_ppo / bc_smoke CLIs."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "reproduce_final200", ROOT / "scripts" / "reproduce_final200.py"
)
repro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repro)


def test_chain_commands_parse_with_current_clis():
    import train_ppo

    stages = repro.load_stages()
    assert [s["name"] for s in stages][0] == "bc" and stages[-1]["name"] == "final200"
    parser = train_ppo.build_parser()
    out = Path("runs")
    prev = "bc"
    for cfg in stages[1:]:
        for smoke in (False, True):
            cmd = repro.ppo_command(cfg, out, smoke)
            args = parser.parse_args(cmd[1:])  # drop the script path
            assert args.name == cfg["name"] and args.resume
            assert args.init_checkpoint.endswith(
                f"{prev}/model.pt" if prev == "bc" else f"{prev}/latest.pt"
            )
            built = train_ppo.make_config(args)
            if not smoke:  # every recorded value survives the round trip through the CLI
                for k, v in cfg.items():
                    if k in repro.REMOVED_KEYS or k in ("out_dir", "init_checkpoint"):
                        continue
                    assert getattr(built, k) == v, (cfg["name"], k)
        prev = cfg["name"]
