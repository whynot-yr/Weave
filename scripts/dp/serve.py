"""Serve a WEAVE DP checkpoint on loopback HTTP, independently of Isaac Sim."""

import argparse
from contextlib import suppress
from pathlib import Path

from weave_data.dp_http_server import DPService, make_server
from weave_data.dp_policy import DPReferencePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"} or not 1 <= args.port <= 65535:
        parser.error("Use a loopback host and port in [1,65535]")
    predictor = DPReferencePredictor(args.checkpoint, args.device)
    service = DPService(predictor, str(args.checkpoint.resolve()))
    with make_server(service, args.host, args.port) as server:
        print(f"DP ready: http://{args.host}:{server.server_port} model={service.health['model_id']}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
