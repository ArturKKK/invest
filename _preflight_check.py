#!/usr/bin/env python3
"""
Pre-flight environment check.
Run before any research/training script to verify package versions match requirements.txt.
Fails fast with clear error if versions mismatch.
"""
import sys
from pathlib import Path

REQUIRED = {
    "numpy": "2.4.3",
    "pandas": "2.3.3",
    "scipy": "1.17.1",
    "lightgbm": "4.6.0",
    "xgboost": "3.2.0",
    "scikit-learn": "1.8.0",
}

def check_versions():
    errors = []
    for pkg, expected in REQUIRED.items():
        import_name = pkg.replace("-", "_")
        if import_name == "scikit_learn":
            import_name = "sklearn"
        try:
            mod = __import__(import_name)
            actual = mod.__version__
            if actual != expected:
                errors.append(f"  {pkg}: expected {expected}, got {actual}")
        except ImportError:
            errors.append(f"  {pkg}: NOT INSTALLED (need {expected})")
    if errors:
        print("=" * 60)
        print("  ❌ PACKAGE VERSION MISMATCH — results will be WRONG")
        print("=" * 60)
        for e in errors:
            print(e)
        print()
        print("Fix: pip install " + " ".join(f"'{k}=={v}'" for k, v in REQUIRED.items()))
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"  ✅ All {len(REQUIRED)} package versions OK")

if __name__ == "__main__":
    check_versions()
