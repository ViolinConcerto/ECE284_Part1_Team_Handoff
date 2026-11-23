"""Train, resume, evaluate and export the ECE284 Part1 W4A4 VGG16.

Run `python train.py --help`. No measured accuracy is bundled with this code.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import QuantConv2d, build_model


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU notebook/kernel or --device cpu.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this machine.")
    return device


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Float32 verification should not silently use TF32 convolution.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def cifar_loaders(args, device):
    # Lazy import keeps numerical tests independent of dataset installation.
    from torchvision import datasets, transforms

    mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_data = datasets.CIFAR10(args.data_root, train=True,
                                  download=args.download, transform=train_transform)
    test_data = datasets.CIFAR10(args.data_root, train=False,
                                 download=args.download, transform=test_transform)
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, test_loader, test_data, generator


def parameter_groups(model, weight_decay):
    clipping, normal = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (clipping if "raw_" in name and "alpha" in name else normal).append(parameter)
    return [dict(params=normal, weight_decay=weight_decay),
            dict(params=clipping, weight_decay=0.0)]


def train_epoch(model, loader, optimizer, device, alpha_regularization=0.0):
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        if alpha_regularization:
            loss = loss + alpha_regularization * sum(
                module.act_alpha.square() for module in model.modules()
                if isinstance(module, QuantConv2d)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss. Last checkpoint has not been overwritten.")
        loss.backward()
        optimizer.step()
        total += labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        loss_sum += loss.detach().item() * labels.numel()
    return dict(loss=loss_sum / total, accuracy_pct=100.0 * correct / total,
                correct=correct, total=total)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    if total == 0:
        raise ValueError("Empty evaluation loader")
    return dict(loss=loss_sum / total, accuracy_pct=100.0 * correct / total,
                correct=correct, total=total)


def atomic_torch_save(payload, target):
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path, device="cpu"):
    # All saved payloads use tensors and basic Python containers only.
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "ece284_part1_v1":
        raise ValueError("Checkpoint is not from this package; explicit architecture conversion is required.")
    return payload


def scheduler_state_for_checkpoint(scheduler):
    """Store milestone multiplicities without pickling collections.Counter.

    Older PyTorch weights-only loaders reject Counter even though recent
    versions allow it. A plain epoch->count dictionary preserves duplicates.
    """
    state = dict(scheduler.state_dict())
    if "milestones" in state:
        state["milestones"] = dict(state["milestones"])
    return state


def restore_scheduler_state(scheduler, state):
    """Restore MultiStepLR's Counter behavior, including closed-form step()."""
    restored = dict(state)
    if "milestones" in restored:
        restored["milestones"] = Counter(restored["milestones"])
    scheduler.load_state_dict(restored)


def make_checkpoint(model, optimizer, scheduler, *, phase, epoch, best_accuracy,
                    args, generator, validation):
    return dict(format="ece284_part1_v1", model_state=model.state_dict(),
                architecture=model.architecture_config(), phase=phase, epoch=epoch,
                quantization_enabled=phase == "qat", best_accuracy_pct=best_accuracy,
                validation=validation, optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler_state_for_checkpoint(scheduler), config=vars(args),
                rng_state=torch.get_rng_state(), loader_rng_state=generator.get_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python_rng_state=random.getstate(),
                numpy_rng_state={"bit_generator": np.random.get_state()[0],
                                 "keys": np.random.get_state()[1].tolist(),
                                 "position": np.random.get_state()[2],
                                 "has_gauss": np.random.get_state()[3],
                                 "cached_gaussian": np.random.get_state()[4]})


def restore_rng(checkpoint, generator):
    torch.set_rng_state(checkpoint["rng_state"].cpu())
    generator.set_state(checkpoint["loader_rng_state"].cpu())
    random.setstate(checkpoint["python_rng_state"])
    state = checkpoint["numpy_rng_state"]
    np.random.set_state((state["bit_generator"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))
    if torch.cuda.is_available() and checkpoint["cuda_rng_states"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_states"]])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify_result(model, test_data, output_dir, args, validation, checkpoint_path):
    from verify_export import verify_and_export

    if not 0 <= args.sample_index < len(test_data):
        raise ValueError("--sample-index lies outside the test dataset")
    image, label = test_data[args.sample_index]
    provenance = dict(evidence_type="trained_model", dataset="CIFAR10", split="test",
                      dataset_index=args.sample_index, label=int(label),
                      checkpoint=str(checkpoint_path), validation=validation,
                      note="Fixed test index; the sample was not selected by verification error.")
    verification = verify_and_export(model, image.unsqueeze(0), output_dir / "vectors",
                                     provenance=provenance)
    success = validation["accuracy_pct"] > 90.0 and verification["formal_pass"]
    summary = dict(status="PASS" if success else "NOT_YET_PASS", evidence_type="trained_model",
                   dataset="CIFAR10", test_accuracy_pct=validation["accuracy_pct"],
                   correct=validation["correct"], total=validation["total"],
                   accuracy_gt_90=validation["accuracy_pct"] > 90.0,
                   max_abs_error=verification["max_abs_error"],
                   error_lt_1e_5=verification["strict_pass"],
                   error_lt_1e_3=verification["formal_pass"],
                   architecture=model.architecture_config(), checkpoint=str(checkpoint_path),
                   vectors="vectors/", note="Strict and formal error limits are reported separately.")
    write_json(output_dir / "results.json", summary)
    report = (
        "Part1: VGG16 with quantization-aware training\n\n"
        "The CIFAR-10 VGG16 contains 13 W4A4 convolution layers. Its selected "
        "features[27] convolution has eight input and eight output channels, a "
        "3x3 kernel, stride 1, padding 1, and no bias. Its following BatchNorm "
        "is removed, leaving ReLU before the next convolution. Signed 4-bit "
        "activations preserve the normalized RGB input in the first layer; "
        "later ReLU activations use unsigned 4-bit codes. Weights use the "
        "course mean/std standardization and learned symmetric clipping.\n\n"
        f"The reloaded best QAT checkpoint achieved {validation['accuracy_pct']:.4f}% "
        f"accuracy ({validation['correct']}/{validation['total']}) on the test split. "
        "The checkpoint was selected using test accuracy, following the course "
        "reference protocol.\n\n"
        "An independent int64 convolution and ReLU were rescaled and compared "
        "against the actual features[29] forward pre-hook input. Captured tensors "
        "were detached and cloned. "
        f"The maximum absolute error was {verification['max_abs_error']:.10g}; "
        f"error <1e-5: {verification['strict_pass']}; error <1e-3: {verification['formal_pass']}. "
        f"The combined formal project criterion was {summary['status']}. "
        "This comparison checks the selected layer's numerical reconstruction; "
        "it is separate from classification accuracy and RTL verification.\n"
    )
    (output_dir / "part1_report.txt").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if success else 2


def smoke_test(args):
    """Check real gradients and reconstruction, without claiming dataset accuracy."""
    from verify_export import verify_and_export

    set_seed(args.seed)
    device = choose_device(args.device)
    model = build_model(quantize=True).to(device)
    images = torch.rand(2, 3, 32, 32, device=device)
    targets = torch.tensor([0, 1], device=device)
    optimizer = torch.optim.SGD(parameter_groups(model, args.weight_decay), lr=0.001)
    model.train()
    loss = nn.functional.cross_entropy(model(images), targets)
    loss.backward()
    grad = model.features[27].weight.grad
    if grad is None or not torch.isfinite(grad).all() or not bool(grad.abs().sum() > 0):
        raise AssertionError("Squeezed layer has no finite nonzero training gradient")
    optimizer.step()
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = verify_and_export(model, images[:1], output / "synthetic_vectors",
                              provenance={"evidence_type": "synthetic_smoke_test",
                                          "note": "Random model/input after ONE synthetic optimizer step. Not trained accuracy evidence."})
    results = dict(evidence_type="synthetic_smoke_test", device=str(device),
                   torch_version=str(torch.__version__), training_step_loss=float(loss.item()),
                   finite_nonzero_conv_gradient=True,
                   max_abs_error=report["max_abs_error"], strict_pass=report["strict_pass"],
                   formal_pass=report["formal_pass"],
                   accuracy_measured=False,
                   note="This test does NOT establish CIFAR10 accuracy >90%.")
    write_json(output / "smoke_test.json", results)
    print(json.dumps(results, indent=2))
    return 0 if report["formal_pass"] else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps or cpu")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="runs/part1")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=284)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--qat-epochs", type=int, default=200)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Optional FP warmup. Default trains QAT from scratch, following the course recipe.")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--warmup-lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha-regularization", type=float, default=1e-5)
    p.add_argument("--milestones", type=int, nargs="+", default=[80, 140, 180])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--resume", type=str, help="Resume last.pth, preserving epoch/optimizer/scheduler and random states")
    p.add_argument("--evaluate", type=str, metavar="CHECKPOINT", help="Evaluate a package checkpoint and export vectors")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid thread/batch-size/worker count")
    if args.qat_epochs < 1 or args.warmup_epochs < 0:
        raise ValueError("Need at least one QAT epoch and nonnegative warmup epochs")
    torch.set_num_threads(args.threads)
    if args.smoke_test:
        return smoke_test(args)
    device = choose_device(args.device)
    if device.type == "cpu":
        print("CPU selected. Full VGG16 QAT is slow; use DSMLP CUDA for the full run.", flush=True)
    set_seed(args.seed, args.deterministic)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and args.evaluate:
        raise ValueError("Choose either --resume or --evaluate")
    if not args.resume and not args.evaluate and (output / "last.pth").exists():
        raise FileExistsError("Run already exists. Use --resume runs/part1/last.pth or choose another --output.")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Use the original checkpoint directory as --output when resuming; it contains best.pth and history.csv.")
    train_loader, test_loader, test_data, generator = cifar_loaders(args, device)
    model = build_model(quantize=True).to(device)
    if args.evaluate:
        checkpoint = load_checkpoint(args.evaluate, device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if not checkpoint["quantization_enabled"]:
            raise ValueError("This is a warmup checkpoint, not a W4A4 QAT result.")
        model.set_quantization(True)
        validation = evaluate(model, test_loader, device)
        return verify_result(model, test_data, output, args, validation, args.evaluate)

    resume = load_checkpoint(args.resume, "cpu") if args.resume else None
    if resume:
        # For a reproducible continuation these choices cannot silently change.
        resume_keys = ("seed", "batch_size", "workers", "threads", "deterministic", "warmup_epochs", "qat_epochs", "lr", "warmup_lr",
                       "weight_decay", "alpha_regularization", "milestones", "gamma")
        changed = [key for key in resume_keys if vars(args)[key] != resume["config"][key]]
        if changed:
            raise ValueError("Resume configuration differs: " + ", ".join(changed) +
                             ". Reuse the original training arguments.")
        model.load_state_dict(resume["model_state"], strict=True)
    config = dict(vars(args), device_selected=str(device), torch_version=str(torch.__version__),
                  architecture=model.architecture_config(),
                  evaluation_protocol="Complete CIFAR10 test split (10000 images). Best checkpoint selected by test accuracy, matching course reference; not an untouched research holdout.")
    write_json(output / "config.json", config)
    (output / "architecture.txt").write_text(str(model) + "\n")
    phases = [("warmup", args.warmup_epochs, args.warmup_lr), ("qat", args.qat_epochs, args.lr)]
    best_qat = float(resume["best_accuracy_pct"]) if resume and resume["phase"] == "qat" else -1.0
    history_path = output / "history.csv"
    fields = ["phase", "epoch", "learning_rate", "train_loss", "train_accuracy_pct",
              "test_loss", "test_accuracy_pct", "seconds"]
    for phase, epochs, lr in phases:
        if epochs == 0:
            continue
        if resume and resume["phase"] == "qat" and phase == "warmup":
            continue
        model.set_quantization(phase == "qat")
        optimizer = torch.optim.SGD(parameter_groups(model, args.weight_decay), lr=lr, momentum=0.9)
        milestones = args.milestones if phase == "qat" else [max(1, int(epochs * 0.7))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
        start = 0
        if resume and phase == resume["phase"]:
            optimizer.load_state_dict(resume["optimizer_state"])
            # torch optimizer load casts states to parameter device.
            restore_scheduler_state(scheduler, resume["scheduler_state"])
            start = resume["epoch"]
            restore_rng(resume, generator)
            resume = None
        for epoch in range(start, epochs):
            started = time.monotonic()
            current_lr = optimizer.param_groups[0]["lr"]
            training = train_epoch(model, train_loader, optimizer, device,
                                   args.alpha_regularization if phase == "qat" else 0.0)
            validation = evaluate(model, test_loader, device)
            scheduler.step()
            improved = phase == "qat" and validation["accuracy_pct"] > best_qat
            if improved:
                best_qat = validation["accuracy_pct"]
            checkpoint = make_checkpoint(model, optimizer, scheduler, phase=phase, epoch=epoch + 1,
                                         best_accuracy=best_qat, args=args, generator=generator,
                                         validation=validation)
            atomic_torch_save(checkpoint, output / "last.pth")
            if improved:
                atomic_torch_save(checkpoint, output / "best.pth")
            record = dict(phase=phase, epoch=epoch + 1, learning_rate=current_lr,
                          train_loss=training["loss"], train_accuracy_pct=training["accuracy_pct"],
                          test_loss=validation["loss"], test_accuracy_pct=validation["accuracy_pct"],
                          seconds=time.monotonic() - started)
            new = not history_path.exists()
            with history_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if new:
                    writer.writeheader()
                writer.writerow(record)
            print(f"{phase} epoch {epoch + 1}/{epochs}: train={training['accuracy_pct']:.2f}% "
                  f"test={validation['accuracy_pct']:.2f}% best_QAT={best_qat:.2f}% "
                  f"lr={current_lr:.6g} time={record['seconds']:.1f}s", flush=True)
    best_path = output / "best.pth"
    if not best_path.exists():
        raise FileNotFoundError("No best QAT checkpoint. Resume in the original --output directory.")
    best = load_checkpoint(best_path, device)
    model.load_state_dict(best["model_state"], strict=True)
    model.set_quantization(True)
    validation = evaluate(model, test_loader, device)
    return verify_result(model, test_data, output, args, validation, best_path)


if __name__ == "__main__":
    sys.exit(main())
