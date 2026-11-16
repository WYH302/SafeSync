"""Additional full-history control aligned to the 30-dB evidence target.

Preserves the original 25-dB full-cache result. This control was added after
the first confirmation analysis to address its budget underuse; it is not
misrepresented as part of the original untouched confirmation specification.
"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_layered_sequences as runner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate-psnr", type=int, choices=[30], default=30)
    args, remaining = parser.parse_known_args()
    if "--methods" in remaining:
        raise ValueError("This fixed control only runs full_cache_mse")
    original_config = runner.config_for
    original_spec = runner.launch_spec

    def config(method):
        if method != "full_cache_mse":
            raise ValueError("Unexpected method in full-history control")
        value = original_config(method)
        value["gate_psnr"] = args.gate_psnr
        return value

    def spec(arguments, manifest):
        value = original_spec(arguments, manifest)
        value.update(schema="additional-full-cache-control-v1",
            additional_control="exact RGB reuse gate aligned to 30-dB evidence target",
            development_history="added after the initial eight-area transmission confirmation; no main-method/teacher changes",
            control_gate_psnr=args.gate_psnr)
        value["thresholds"]["gate_psnr"] = args.gate_psnr
        return value

    runner.config_for = config
    runner.launch_spec = spec
    runner.SOURCES = [*runner.SOURCES, Path(__file__).relative_to(ROOT).as_posix()]
    sys.argv = [str(__file__), *remaining, "--methods", "full_cache_mse"]
    runner.main()


if __name__ == "__main__":
    main()
