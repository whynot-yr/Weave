"""Single-device or torchrun DDP reference-flow training with exact masked loss."""

import argparse
from contextlib import nullcontext
import hashlib
from itertools import islice
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from _common import read_config, write_new_json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from weave_policy.data.normalization import load_statistics
from weave_policy.data.reference_dataset import ReferenceWindowDataset
from weave_policy.data.split import load_split
from weave_policy.flow_matching_policy import FlowMatchingPolicy
from weave_policy.models.dinov2_encoder import DinoV2Encoder
from weave_policy.training.distributed import (
    DistributedContext,
    DistributedEvaluationSampler,
    backward_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--features", help="Full DINOv2 cache; otherwise encode RGB during training")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", help="DDP CUDA index is assigned by LOCAL_RANK")
    parser.add_argument("--steps", type=int, default=100000, help="Total optimizer updates")
    parser.add_argument("--batch-size", type=int, default=32, help="Microbatch size PER RANK")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers PER RANK")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--amp", action="store_true", help="CUDA bf16 autocast; loss reduction stays FP32")
    parser.add_argument("--resume", type=Path, help="Exact training resume; same world size and batch protocol")
    parser.add_argument("--init-checkpoint", type=Path, help="Load weights only; start a new optimizer/run")
    parser.add_argument("--smoke", action="store_true", help="2 blocks / width 96, never a production model")
    parser.add_argument("--dino-repo")
    parser.add_argument("--dino-weights")
    parser.add_argument("--dino-ref", default="main")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.eval_every, args.gradient_accumulation_steps) <= 0:
        parser.error("Steps/batch size/intervals/accumulation must be positive")
    if args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0 or args.warmup_steps < 0:
        parser.error("Invalid workers, learning rate, or warmup")
    if args.amp and not args.device.startswith("cuda"):
        parser.error("--amp requires CUDA")
    if args.resume and args.init_checkpoint:
        parser.error("Choose --resume or --init-checkpoint")
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def rng_state(device):
    numpy_state = np.random.get_state()
    return {
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["rng"])
    if device.type == "cuda" and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    if "python_rng" in state:
        random.setstate(state["python_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def train_run(args, context):
    device = context.device
    seed_all(args.seed)
    config = read_config(args.config)
    if args.smoke:
        config = replace(config, width=96, num_heads=2, head_dim=48, num_layers=2)
    split = load_split(args.split, args.source)
    train = ReferenceWindowDataset(args.source, split["train_episodes"], config.reference_offsets, args.features)
    evaluation = ReferenceWindowDataset(args.source, split["val_episodes"], config.reference_offsets, args.features)
    metrics = None
    try:
        if abs(train.info["metadata"]["fps"] - config.fps) > 1e-6:
            raise ValueError("Source FPS differs from the model reference protocol")
        stats = load_statistics(args.stats, train)
        encoder = None
        if args.features:
            encoder_meta = train.feature_metadata
        else:
            # Rank zero finishes Hub loading/downloads before other ranks load.
            def load_encoder():
                nonlocal encoder
                encoder = DinoV2Encoder.from_pretrained(args.dino_repo, args.dino_weights, args.dino_ref).to(device)
                return encoder.metadata()

            encoder_meta = context.rank_zero_call(load_encoder)
            if context.rank != 0:
                encoder_meta = load_encoder()
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
            "world_size": context.world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "sampler": "DistributedSampler:shuffle=True:drop_last=False",
            "loss_normalization": "global_valid_elements_per_optimizer_update",
            "training_state_version": 2,
        }
        signature = json.loads(json.dumps(signature))
        context.require_same(signature, "data/model/encoder/training protocol")
        # Paths/budgets must agree as well, especially collective trigger steps.
        context.require_same(
            [str(args.output.resolve()), str(args.resume), str(args.init_checkpoint), args.save_every, args.eval_every],
            "output/checkpoints/collective intervals",
        )
        metadata = {"training_signature": signature, "source_info": train.info, "encoder": encoder_meta}
        checkpoint = args.resume or args.init_checkpoint
        legacy_resume = False
        if checkpoint:
            policy, saved = FlowMatchingPolicy.from_checkpoint(checkpoint, device)
            if args.resume:
                previous = saved["training_signature"]
                if "training_state_version" not in previous:
                    legacy_resume = True
                    legacy_keys = set(signature) - {
                        "world_size", "gradient_accumulation_steps", "sampler",
                        "loss_normalization", "training_state_version",
                    }
                    matches = (
                        context.world_size == 1 and args.gradient_accumulation_steps == 1
                        and previous == {key: signature[key] for key in legacy_keys}
                    )
                else:
                    matches = previous == signature
                if not matches:
                    raise ValueError("Resume protocol mismatch (data/model/budget/world size/batch/seed/optimizer)")
            elif (
                saved["training_signature"]["config"] != signature["config"]
                or saved["training_signature"]["stats_sha256"] != signature["stats_sha256"]
                or saved["encoder"]["encoder_id"] != encoder_meta["encoder_id"]
            ):
                raise ValueError("Initialization checkpoint model/normalization/encoder mismatch")
        else:
            policy = FlowMatchingPolicy(config, stats).to(device)
        model = DDP(policy, device_ids=[device.index] if device.type == "cuda" else None) if context.distributed else policy
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)

        def schedule(step):
            if step < args.warmup_steps:
                return (step + 1) / max(args.warmup_steps, 1)
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        sampler = DistributedSampler(
            train, num_replicas=context.world_size, rank=context.rank, seed=args.seed,
            shuffle=True, drop_last=False,
        )
        batches_per_epoch = math.ceil(len(sampler) / args.batch_size)
        step, epoch, next_batch = 0, 0, 0
        seed_all(args.seed + context.rank)
        resume_rng = None
        if args.resume:
            state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = state["step"]
            if legacy_resume:
                epoch, next_batch = divmod(step, batches_per_epoch)
                legacy_cuda = state["cuda_rng"]
                resume_rng = {"rng": state["rng"], "cuda_rng": legacy_cuda[device.index] if legacy_cuda else None}
            else:
                epoch, next_batch = state["epoch"], state["next_batch"]
                resume_rng = state["rank_rng_states"][context.rank]
        if step >= args.steps:
            raise ValueError("Checkpoint already reached the requested training budget")
        if not 0 <= next_batch < batches_per_epoch:
            raise ValueError("Invalid saved batch cursor")

        def create_output():
            nonlocal metrics
            args.output.mkdir(parents=True, exist_ok=False)
            arguments = vars(args) | {
                "output": str(args.output), "resume": str(args.resume), "init_checkpoint": str(args.init_checkpoint),
                "world_size": context.world_size,
                "effective_global_batch_size": args.batch_size * context.world_size * args.gradient_accumulation_steps,
            }
            write_new_json(args.output / "run_args.json", arguments)
            metrics = (args.output / "metrics.jsonl").open("x")

        context.rank_zero_call(create_output)
        if context.rank == 0:
            print(json.dumps({
                "world_size": context.world_size, "local_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_global_batch_size": args.batch_size * context.world_size * args.gradient_accumulation_steps,
                "sampler_repeated_positions_per_epoch": len(sampler) * context.world_size - len(train),
            }), flush=True)
        policy.train()
        loader_options = {
            "num_workers": args.workers,
            "pin_memory": device.type == "cuda",
        }
        if args.workers:
            # HDF5 handles are opened inside each worker, never inherited by fork.
            loader_options["multiprocessing_context"] = "spawn"

        def forward(batch, module):
            tokens = batch["image_tokens"] if encoder is None else encoder(batch["rgb"])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                return module(tokens, batch["reference"], batch["reference_mask"])

        def evaluate():
            policy.eval()
            total = torch.zeros((), device=device, dtype=torch.float64)
            count = torch.zeros((), device=device, dtype=torch.int64)
            devices = [device.index] if device.type == "cuda" else []
            # Independent deterministic validation randomness; preserve training RNG.
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(args.seed + 100000 + context.rank)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(args.seed + 100000 + context.rank)
                val_loader = DataLoader(
                    evaluation, batch_size=args.batch_size,
                    sampler=DistributedEvaluationSampler(evaluation, context.rank, context.world_size),
                    generator=torch.Generator().manual_seed(args.seed + context.rank), **loader_options,
                )
                # Unequal numbers of validation batches are safe: no DDP forwards
                # and no collectives inside this loop.
                for batch in val_loader:
                    output = forward(batch, policy)
                    total += output["loss_sum"].double()
                    count += output["valid_count"]
            total, count = context.sum(total), context.sum(count)
            if count.item() <= 0 or not torch.isfinite(total):
                raise FloatingPointError("Empty or nonfinite global validation loss")
            policy.train()
            return {"val_loss": (total / count).item(), "val_valid_elements": count.item()}

        while step < args.steps:
            sampler.set_epoch(epoch)
            order = list(sampler)
            batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)][next_batch:]
            loader = DataLoader(
                train, batch_sampler=batches,
                generator=torch.Generator().manual_seed(args.seed + epoch * context.world_size + context.rank),
                **loader_options,
            )
            iterator = iter(loader)
            if resume_rng is not None:
                # Restore after model/DDP/loader construction, before first forward.
                restore_rng(resume_rng, device)
                resume_rng = None
            while step < args.steps:
                window = list(islice(iterator, args.gradient_accumulation_steps))
                if not window:
                    break
                local_count = torch.tensor(
                    sum(int(batch["reference_mask"].sum()) for batch in window), device=device, dtype=torch.int64,
                )
                global_count = context.sum(local_count)
                optimizer.zero_grad(set_to_none=True)
                update_sum = torch.zeros((), device=device, dtype=torch.float64)
                for index, batch in enumerate(window):
                    sync = model.no_sync() if context.distributed and index < len(window) - 1 else nullcontext()
                    with sync:
                        output = forward(batch, model)
                        context.require_finite(output["loss_sum"], f"training loss at update {step}")
                        loss = backward_loss(output["loss_sum"], global_count, context.world_size)
                        loss.backward()
                    update_sum += output["loss_sum"].detach().double()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=False)
                context.require_finite(grad_norm, f"gradient norm at update {step}")
                optimizer.step()
                scheduler.step()
                step += 1
                next_batch += len(window)
                if next_batch == batches_per_epoch:
                    epoch += 1
                    next_batch = 0
                global_sum = context.sum(update_sum)
                record = {
                    "step": step, "loss": (global_sum / global_count).item(),
                    "valid_elements": global_count.item(), "microbatches": len(window),
                    "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                }
                if step % args.eval_every == 0:
                    record.update(evaluate())

                def log_record():
                    if step == 1 or step % 100 == 0 or "val_loss" in record:
                        print(json.dumps(record), flush=True)
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()

                context.rank_zero_call(log_record)
                if step % args.save_every == 0 or step == args.steps:
                    rank_states = context.gather_objects(rng_state(device))

                    def save():
                        state = {
                            "version": 2, "step": step, "epoch": epoch, "next_batch": next_batch,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "rank_rng_states": rank_states,
                        }
                        path = args.output / f"step_{step:08d}"
                        policy.save_checkpoint(path, metadata, state)
                        print(f"saved {path}", flush=True)

                    context.rank_zero_call(save)
    finally:
        if metrics is not None:
            metrics.close()
        train.close()
        evaluation.close()


def main():
    args = parse_args()
    context = DistributedContext.initialize(args.device)
    try:
        train_run(args, context)
    finally:
        context.close()


if __name__ == "__main__":
    main()
