"""Enable supported SGLang completed-request dumps without changing training."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    settings = json.loads((root / "settings.json").read_text())
    argv = settings["argv"]
    expected = argv[argv.index("--hf-checkpoint") + 1]
    info_response = requests.get(args.engine_url + "/server_info", timeout=30)
    info_response.raise_for_status()
    info = info_response.json()
    if info["model_path"] != expected:
        raise ValueError("The engine checkpoint does not match this run")
    destination = root / "sglang-requests"
    destination.mkdir(exist_ok=True)
    configuration = dict(
        dump_requests_folder=str(destination),
        dump_requests_threshold=12,
        dump_requests_exclude_meta_keys=[
            "routed_experts",
            "hidden_states",
            "input_token_logprobs",
            "output_token_logprobs",
            "input_top_logprobs",
            "output_top_logprobs",
        ],
    )
    response = requests.post(args.engine_url + "/configure_logging", json=configuration, timeout=30)
    response.raise_for_status()
    record = dict(
        enabled_at=datetime.now(timezone.utc).isoformat(),
        engine_url=args.engine_url,
        configuration=configuration,
        model_path=expected,
        limitation="Earlier requests and a final incomplete 12-request dump may be absent; preserve coverage bounds.",
    )
    (root / "request-cost-logging.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
