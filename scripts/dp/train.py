"""WEAVE data entry to the unmodified LeRobot DiffusionPolicy training API.

Single-device trainer using LeRobot policy, normalization, optimizer/scheduler
presets and save_pretrained. No monkeypatches of lerobot-train or model code.
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from torch.utils.data import DataLoader
from weave_data.dp_codec import IMAGE_KEY
from weave_data.dp_dataset import DPWindowDataset
from weave_data.dp_stats import load_stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-episodes", nargs="+", type=int, required=True)
    parser.add_argument("--eval-episodes", nargs="+", type=int, default=[])
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path, help="Trusted local step checkpoint; steps must remain unchanged")
    parser.add_argument("--smoke", action="store_true", help="Small U-Net, no downloaded vision weights; testing only")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.eval_every) <= 0 or args.workers < 0:
        parser.error("Steps, batch size and intervals must be positive; workers must be nonnegative")
    return args


def split_guard(train, evaluation):
    if set(train.episodes) & set(evaluation.episodes):
        raise ValueError("Train and eval episodes overlap")

    def families(dataset):
        return {
            re.sub(r"_v\d+$", "", e["clip_name"])
            for e in dataset.protocol["episodes"]
            if e["episode_index"] in dataset.episodes
        }

    overlap = families(train) & families(evaluation)
    if overlap:
        raise ValueError(f"Train/eval share original motion families: {sorted(overlap)}")


def save_checkpoint(output, step, policy, pre, post, optimizer, scheduler, signature):
    checkpoint = output / f"step_{step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "signature": signature,
        },
        checkpoint / "training_state.pt",
    )
    (checkpoint / "weave_training.json").write_text(json.dumps(signature, indent=2))
    shutil.copyfile(output / "weave_protocol.json", checkpoint / "weave_protocol.json")
    print(f"Saved {checkpoint}", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train = DPWindowDataset(args.dataset, args.train_episodes)
    stats = load_stats(args.stats, train)
    evaluation = DPWindowDataset(args.dataset, args.eval_episodes) if args.eval_episodes else None
    if evaluation:
        split_guard(train, evaluation)
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}")
    signature = {
        "dataset_id": train.protocol["dataset_id"],
        "train_episodes": train.episodes,
        "eval_episodes": sorted(args.eval_episodes),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "horizon": train.horizon,
        "smoke": args.smoke,
    }
    if args.resume:
        saved = json.loads((args.resume / "weave_training.json").read_text())
        if saved != signature:
            raise ValueError("Resume dataset/split/schedule/batch size/seed/model does not match checkpoint")
        cfg = DiffusionConfig.from_pretrained(args.resume)
        cfg.device = args.device
        # Weights are in the checkpoint, no ImageNet download needed on resume.
        cfg.pretrained_backbone_weights = None
        policy = DiffusionPolicy.from_pretrained(args.resume, config=cfg)
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=str(args.resume), preprocessor_overrides={"device_processor": {"device": args.device}}
        )
    else:
        cfg = DiffusionConfig(
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (88,)),
                IMAGE_KEY: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (125,))},
            device=args.device,
            n_obs_steps=1,
            horizon=40,
            n_action_steps=40,
            drop_n_last_frames=0,
            do_mask_loss_for_padding=True,
            down_dims=(32, 64, 128) if args.smoke else (512, 1024, 2048),
            pretrained_backbone_weights=None if args.smoke else "ResNet18_Weights.IMAGENET1K_V1",
            num_inference_steps=2 if args.smoke else 100,
        )
        policy = DiffusionPolicy(cfg)
        pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.to(args.device).train()
    optim_cfg = cfg.get_optimizer_preset()
    optimizer = optim_cfg.build(policy.get_optim_params())
    scheduler = cfg.get_scheduler_preset().build(optimizer, args.steps)
    step = 0
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        torch.set_rng_state(state["rng"])
        if args.device.startswith("cuda") and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "weave_protocol.json").write_text(json.dumps(train.protocol, indent=2))
    (output / "run_args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    # Deterministic per-epoch permutations allow resuming at the same next batch.
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    while step < args.steps:
        epoch, first_batch = divmod(step, batches_per_epoch)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)][first_batch:]
        loader = DataLoader(
            train,
            batch_sampler=batches,
            num_workers=args.workers,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            pin_memory=args.device.startswith("cuda"),
        )
        for batch in loader:
            loss, _ = policy(pre(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), optim_cfg.grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"step={step} loss={loss.item():.6f}", flush=True)
            if evaluation and step % args.eval_every == 0:
                policy.eval()
                total, count = 0.0, 0
                # Evaluation's diffusion noise must not alter the training RNG.
                devices = [torch.device(args.device).index or 0] if args.device.startswith("cuda") else []
                with torch.no_grad(), torch.random.fork_rng(devices=devices):
                    for batch in DataLoader(evaluation, batch_size=args.batch_size, num_workers=args.workers):
                        n = len(batch["action"])
                        val_loss, _ = policy(pre(batch))
                        total += val_loss.item() * n
                        count += n
                print(f"step={step} eval_loss={total / count:.6f}", flush=True)
                policy.train()
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(output, step, policy, pre, post, optimizer, scheduler, signature)
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
