#!/usr/bin/env python3
"""
Model Registry — track & archive every production model generation.

Usage:
  # Register current prod models (auto-detects all results_*_prod dirs)
  python model_registry.py register --tag "v2.1-calendar" --notes "Added 9 calendar features"

  # Archive current prod before overwriting (copies to models_archive/gen_005_...)
  python model_registry.py archive

  # List all registered generations
  python model_registry.py list

  # Restore a specific generation
  python model_registry.py restore --gen 3

  # Show details of a generation
  python model_registry.py show --gen 3

  # Compare two generations
  python model_registry.py diff --gen1 3 --gen2 4

Integration with training scripts:
  Called automatically at the end of train_production.sh (archive + register).
  Can also be called manually before/after any retrain.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
REGISTRY_FILE = ROOT / "model_registry.json"
ARCHIVE_DIR = ROOT / "models_archive"

# All prod model directories we track
PROD_DIRS = [
    "results_v6_prod",
    "results_v7_prod",
    "results_catboost_prod",
    "results_xgboost_prod",
    "results_mlp_prod",
]

# Extensions per model type
MODEL_EXTENSIONS = {
    "results_v6_prod": ".txt",
    "results_v7_prod": ".txt",
    "results_catboost_prod": ".cbm",
    "results_xgboost_prod": ".json",
    "results_mlp_prod": ".pt",
}


def load_registry():
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"generations": [], "current_gen": 0}


def save_registry(reg):
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, ensure_ascii=False))


def scan_prod_dir(dirpath: Path) -> dict:
    """Scan a prod directory → metadata dict."""
    if not dirpath.exists():
        return None

    info = {"path": str(dirpath.relative_to(ROOT)), "exists": True}

    # Feature names
    fn_path = dirpath / "feature_names.json"
    if fn_path.exists():
        feats = json.loads(fn_path.read_text())
        info["n_features"] = len(feats)
        info["has_calendar"] = any(f.startswith("cal_") for f in feats)
        info["has_derivatives"] = any(f.startswith("deriv_") or f.startswith("oi_") for f in feats)
        info["has_sentiment"] = any("news" in f or "sentiment" in f for f in feats)
    else:
        info["n_features"] = 0

    # Production meta
    meta_path = dirpath / "production_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        info["train_end"] = meta.get("train_end")
        info["val_end"] = meta.get("val_end")
        info["train_rows"] = meta.get("train_rows")
        info["trained_at"] = meta.get("timestamp")

    # Model files
    ext = MODEL_EXTENSIONS.get(dirpath.name, ".*")
    model_files = sorted(dirpath.glob(f"*{ext}")) if ext != ".*" else []
    # For xgboost, skip feature_names.json (also .json)
    if ext == ".json":
        model_files = [f for f in model_files if "model_seed" in f.name]
    info["n_models"] = len(model_files)
    info["model_files"] = [f.name for f in model_files]

    # File dates
    if model_files:
        mod_times = [f.stat().st_mtime for f in model_files]
        info["model_date"] = datetime.fromtimestamp(max(mod_times)).isoformat()

    # Total size
    total_bytes = sum(f.stat().st_size for f in dirpath.rglob("*") if f.is_file())
    info["size_mb"] = round(total_bytes / 1024 / 1024, 1)

    return info


def cmd_register(args):
    """Register current prod models as a generation."""
    reg = load_registry()
    gen_num = reg["current_gen"] + 1

    gen = {
        "gen": gen_num,
        "tag": args.tag or f"gen_{gen_num:03d}",
        "timestamp": datetime.now().isoformat(),
        "notes": args.notes or "",
        "models": {},
    }

    print(f"📋 Registering generation #{gen_num}: {gen['tag']}")

    for dname in PROD_DIRS:
        dirpath = ROOT / dname
        info = scan_prod_dir(dirpath)
        if info:
            gen["models"][dname] = info
            cal = "✅ cal" if info.get("has_calendar") else "❌ no-cal"
            print(f"   {dname}: {info['n_models']} models, "
                  f"{info['n_features']} feats, {cal}, {info['size_mb']}MB")
        else:
            print(f"   {dname}: (not found)")

    reg["generations"].append(gen)
    reg["current_gen"] = gen_num
    save_registry(reg)
    print(f"\n✅ Registered as gen #{gen_num} ({gen['tag']})")
    return gen_num


def cmd_archive(args):
    """Archive current prod models before overwriting."""
    reg = load_registry()
    gen_num = reg["current_gen"]

    if gen_num == 0:
        # Auto-register first
        print("   No generations registered yet, registering current state...")
        args.tag = args.tag if hasattr(args, 'tag') and args.tag else None
        args.notes = args.notes if hasattr(args, 'notes') and args.notes else "Auto-registered before archive"
        gen_num = cmd_register(args)

    gen = reg["generations"][-1]
    tag = gen["tag"]
    archive_name = f"gen_{gen_num:03d}_{tag}_{datetime.now().strftime('%Y%m%d')}"
    archive_path = ARCHIVE_DIR / archive_name

    print(f"\n📦 Archiving gen #{gen_num} → {archive_path.relative_to(ROOT)}")
    archive_path.mkdir(parents=True, exist_ok=True)

    for dname in PROD_DIRS:
        src = ROOT / dname
        if src.exists():
            dst = archive_path / dname
            shutil.copytree(src, dst, dirs_exist_ok=True)
            size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1024 / 1024
            print(f"   {dname} → {size_mb:.1f} MB")

    # Update registry with archive path
    gen["archive_path"] = str(archive_path.relative_to(ROOT))
    save_registry(reg)

    total_mb = sum(f.stat().st_size for f in archive_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"\n✅ Archived ({total_mb:.1f} MB total)")


def cmd_restore(args):
    """Restore a specific generation from archive."""
    reg = load_registry()
    gen = next((g for g in reg["generations"] if g["gen"] == args.gen), None)
    if not gen:
        print(f"❌ Generation #{args.gen} not found")
        return

    archive_path = gen.get("archive_path")
    if not archive_path:
        print(f"❌ Generation #{args.gen} has no archive")
        return

    archive_path = ROOT / archive_path
    if not archive_path.exists():
        print(f"❌ Archive dir not found: {archive_path}")
        return

    print(f"🔄 Restoring gen #{args.gen} ({gen['tag']}) from {archive_path.relative_to(ROOT)}")

    if not args.force:
        resp = input("   This will OVERWRITE current prod dirs. Continue? [y/N] ")
        if resp.lower() != 'y':
            print("   Aborted")
            return

    for dname in PROD_DIRS:
        src = archive_path / dname
        dst = ROOT / dname
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"   ✅ {dname} restored")

    print(f"\n✅ Restored gen #{args.gen}")


def cmd_list(args):
    """List all registered generations."""
    reg = load_registry()
    if not reg["generations"]:
        print("No generations registered. Run: python model_registry.py register")
        return

    print(f"{'Gen':>4} {'Tag':<25} {'Date':<20} {'Models':>6} {'Feats':>5} {'Cal':>4} {'Archive':>8}")
    print("─" * 80)

    for gen in reg["generations"]:
        n_models = sum(m.get("n_models", 0) for m in gen["models"].values())
        feats = set()
        has_cal = False
        for m in gen["models"].values():
            feats.add(m.get("n_features", 0))
            if m.get("has_calendar"):
                has_cal = True

        feats_str = "/".join(str(f) for f in sorted(feats)) if feats else "?"
        date_str = gen["timestamp"][:16]
        archived = "✅" if gen.get("archive_path") else "—"
        cal_str = "✅" if has_cal else "❌"

        print(f"{gen['gen']:>4} {gen['tag']:<25} {date_str:<20} {n_models:>6} "
              f"{feats_str:>5} {cal_str:>4} {archived:>8}")

    if reg["generations"]:
        print(f"\nCurrent: gen #{reg['current_gen']}")


def cmd_show(args):
    """Show details of a generation."""
    reg = load_registry()
    gen = next((g for g in reg["generations"] if g["gen"] == args.gen), None)
    if not gen:
        print(f"❌ Generation #{args.gen} not found")
        return

    print(f"Generation #{gen['gen']}: {gen['tag']}")
    print(f"  Timestamp: {gen['timestamp']}")
    print(f"  Notes: {gen.get('notes', '—')}")
    print(f"  Archive: {gen.get('archive_path', '—')}")
    print()

    for dname, info in gen.get("models", {}).items():
        print(f"  {dname}:")
        print(f"    Models: {info.get('n_models', 0)} × {MODEL_EXTENSIONS.get(dname, '?')}")
        print(f"    Features: {info.get('n_features', '?')}")
        print(f"    Calendar: {'✅' if info.get('has_calendar') else '❌'}")
        print(f"    Derivatives: {'✅' if info.get('has_derivatives') else '❌'}")
        print(f"    Sentiment: {'✅' if info.get('has_sentiment') else '❌'}")
        if info.get("train_end"):
            print(f"    Train end: {info['train_end']}")
            print(f"    Val end: {info.get('val_end', '?')}")
        if info.get("model_date"):
            print(f"    Model date: {info['model_date']}")
        print(f"    Size: {info.get('size_mb', '?')} MB")


def cmd_diff(args):
    """Compare two generations."""
    reg = load_registry()
    g1 = next((g for g in reg["generations"] if g["gen"] == args.gen1), None)
    g2 = next((g for g in reg["generations"] if g["gen"] == args.gen2), None)
    if not g1 or not g2:
        print(f"❌ Generation not found")
        return

    print(f"Comparing gen #{args.gen1} ({g1['tag']}) vs gen #{args.gen2} ({g2['tag']})")
    print()

    all_dirs = set(list(g1.get("models", {}).keys()) + list(g2.get("models", {}).keys()))
    for dname in sorted(all_dirs):
        m1 = g1.get("models", {}).get(dname, {})
        m2 = g2.get("models", {}).get(dname, {})

        changes = []
        if m1.get("n_features") != m2.get("n_features"):
            changes.append(f"feats: {m1.get('n_features', '—')} → {m2.get('n_features', '—')}")
        if m1.get("has_calendar") != m2.get("has_calendar"):
            changes.append(f"calendar: {m1.get('has_calendar', '—')} → {m2.get('has_calendar', '—')}")
        if m1.get("n_models") != m2.get("n_models"):
            changes.append(f"models: {m1.get('n_models', '—')} → {m2.get('n_models', '—')}")
        if m1.get("train_end") != m2.get("train_end"):
            changes.append(f"train_end: {m1.get('train_end', '—')} → {m2.get('train_end', '—')}")

        if changes:
            print(f"  {dname}: {', '.join(changes)}")
        elif m1 and m2:
            print(f"  {dname}: (no changes)")
        elif not m1:
            print(f"  {dname}: NEW in gen #{args.gen2}")
        else:
            print(f"  {dname}: REMOVED in gen #{args.gen2}")


def main():
    parser = argparse.ArgumentParser(description="Model Registry")
    sub = parser.add_subparsers(dest="cmd")

    p_reg = sub.add_parser("register", help="Register current prod models")
    p_reg.add_argument("--tag", type=str, default=None)
    p_reg.add_argument("--notes", type=str, default=None)

    p_arch = sub.add_parser("archive", help="Archive current prod before overwrite")
    p_arch.add_argument("--tag", type=str, default=None)
    p_arch.add_argument("--notes", type=str, default=None)

    p_list = sub.add_parser("list", help="List all generations")

    p_show = sub.add_parser("show", help="Show generation details")
    p_show.add_argument("--gen", type=int, required=True)

    p_restore = sub.add_parser("restore", help="Restore generation from archive")
    p_restore.add_argument("--gen", type=int, required=True)
    p_restore.add_argument("--force", action="store_true")

    p_diff = sub.add_parser("diff", help="Compare two generations")
    p_diff.add_argument("--gen1", type=int, required=True)
    p_diff.add_argument("--gen2", type=int, required=True)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    {"register": cmd_register, "archive": cmd_archive, "list": cmd_list,
     "show": cmd_show, "restore": cmd_restore, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    main()
