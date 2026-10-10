"""Single-device reference-flow trainer with deterministic batch resume."""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from _common import read_config, write_new_json
from torch.utils.data import DataLoader
from weave_policy.data.normalization import load_statistics
from weave_policy.data.reference_dataset import ReferenceWindowDataset
from weave_policy.data.split import load_split
from weave_policy.flow_matching_policy import FlowMatchingPolicy
from weave_policy.models.dinov2_encoder import DinoV2Encoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--features", help="Full DINOv2 cache; otherwise encode RGB during training")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--amp", action="store_true", help="CUDA bf16 autocast")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true", help="2 blocks / width 96, never a production model")
    parser.add_argument("--dino-repo")
    parser.add_argument("--dino-weights")
    parser.add_argument("--dino-ref", default="main")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.eval_every) <= 0 or args.workers < 0:
        parser.error("Invalid steps/batch size/intervals/workers")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.warmup_steps < 0:
        parser.error("Invalid learning rate or warmup")
    if args.amp and not args.device.startswith("cuda"):
        parser.error("--amp requires CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    config = read_config(args.config)
    if args.smoke:
        config = replace(config, width=96, num_heads=2, head_dim=48, num_layers=2)
    split = load_split(args.split, args.source)
    train = ReferenceWindowDataset(args.source, split["train_episodes"], config.reference_offsets, args.features)
    evaluation = ReferenceWindowDataset(args.source, split["val_episodes"], config.reference_offsets, args.features)
    if abs(train.info["metadata"]["fps"] - config.fps) > 1e-6:
        raise ValueError("Source FPS differs from the model reference protocol")
    stats = load_statistics(args.stats, train)
    encoder = None
    if args.features:
        encoder_meta = train.feature_metadata
    else:
        encoder = DinoV2Encoder.from_pretrained(args.dino_repo, args.dino_weights, args.dino_ref).to(args.device)
        encoder_meta = encoder.metadata()
    signature = {
        "config": config.to_dict(),
        "source_id": train.info["source_id"],
        "split": split,
        "stats_sha256": hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest(),
        "encoder_id": encoder_meta["encoder_id"],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "amp": args.amp,
        "smoke": args.smoke,
    }
    # JSON canonicalization turns tuple offsets into lists for resume comparison.
    signature = json.loads(json.dumps(signature))
    metadata = {"training_signature": signature, "source_info": train.info, "encoder": encoder_meta}
    if args.resume:
        policy, saved = FlowMatchingPolicy.from_checkpoint(args.resume, args.device)
        if saved["training_signature"] != signature:
            raise ValueError("Resume data/normalization/model/budget/batch/seed/optimizer mismatch")
    else:
        policy = FlowMatchingPolicy(config, stats).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)

    def schedule(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step = 0
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        torch.set_rng_state(state["rng"])
        if args.device.startswith("cuda") and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if step >= args.steps:
        raise ValueError("Checkpoint already reached the requested training budget")
    args.output.mkdir(parents=True)
    write_new_json(args.output / "run_args.json", vars(args) | {"output": str(args.output), "resume": str(args.resume)})
    policy.train()
    batches_per_epoch = math.ceil(len(train) / args.batch_size)

    def compute_loss(batch):
        tokens = batch["image_tokens"] if encoder is None else encoder(batch["rgb"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            return policy(tokens, batch["reference"], batch["reference_mask"])["loss"]

    def evaluate():
        policy.eval()
        total, count = 0.0, 0
        devices = [torch.device(args.device).index or 0] if args.device.startswith("cuda") else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            for batch in DataLoader(
                evaluation,
                batch_size=args.batch_size,
                num_workers=args.workers,
                generator=torch.Generator().manual_seed(args.seed),
            ):
                weight = int(batch["reference_mask"].sum())
                loss = compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                total += float(loss) * weight
                count += weight
        policy.train()
        return total / count

    try:
        with (args.output / "metrics.jsonl").open("x") as metrics:
            while step < args.steps:
                epoch, first_batch = divmod(step, batches_per_epoch)
                order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
                batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)][first_batch:]
                loader = DataLoader(
                    train,
                    batch_sampler=batches,
                    num_workers=args.workers,
                    generator=torch.Generator().manual_seed(args.seed + epoch),
                )
                for batch in loader:
                    loss = compute_loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    record = {"step": step, "loss": float(loss.detach()), "lr": scheduler.get_last_lr()[0]}
                    if step % args.eval_every == 0:
                        record["val_loss"] = evaluate()
                    if step == 1 or step % 100 == 0 or "val_loss" in record:
                        print(json.dumps(record), flush=True)
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    if step % args.save_every == 0 or step == args.steps:
                        state = {
                            "step": step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                        }
                        checkpoint = args.output / f"step_{step:08d}"
                        policy.save_checkpoint(checkpoint, metadata, state)
                        print(f"saved {checkpoint}", flush=True)
                    if step >= args.steps:
                        break
    finally:
        train.close()
        evaluation.close()


if __name__ == "__main__":
    main()
