#!/usr/bin/env python3
"""Exercise the configured real model: image -> tool call -> tool result -> completed answer."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_gateway import GatewayError, ModelGateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data'/'demo')
    args = parser.parse_args()
    try:
        result = ModelGateway(args.data_dir/'agent-settings.json').probe()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except GatewayError as error:
        print(json.dumps(error.public(), ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
