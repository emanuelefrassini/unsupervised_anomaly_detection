"""Optional, reviewable compatibility changes for the original single-stream VAE.

This does NOT fix scientific issues such as training/test preprocessing leakage,
or implement the paper's two-stream architecture.

Usage:
    python scripts/patch_legacy_vae.py --check
    python scripts/patch_legacy_vae.py --apply
    git diff -- VAE_reproducible_improved.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "VAE_reproducible_improved.py"

PATH_FIELDS = {
    "normal_feature_folder": ("CAG_NORMAL_FEATURES", "data/normal/features"),
    "normal_annotation_folder": ("CAG_NORMAL_ANNOTATIONS", "data/normal/annotations"),
    "anomaly_feature_folder": ("CAG_ANOMALY_FEATURES", "data/anomaly/features"),
    "anomaly_annotation_folder": ("CAG_ANOMALY_ANNOTATIONS", "data/anomaly/annotations"),
    "base_log_dir": ("CAG_RUNS_DIR", "runs"),
}


def patch_source(original: str) -> tuple[str, list[str]]:
    source = original
    changes: list[str] = []
    for field, (env_name, default) in PATH_FIELDS.items():
        regex = re.compile(rf"(?m)^(?P<indent>[ \t]*)self\.{re.escape(field)}\s*=\s*(?P<expr>[^\n]+)$")
        match = regex.search(source)
        if not match:
            raise RuntimeError(f"Expected Config.{field} assignment not found; no files modified")
        old_expr = match.group("expr")
        if "os.environ.get(" in old_expr:
            continue  # Already configured
        # Do not clobber user-supplied configuration that isn't the historic Windows path.
        if "OneDrive" not in old_expr and "C:\\Users\\" not in old_expr:
            raise RuntimeError(f"Config.{field} is not the known historic path; refusing to replace")
        new = (
            f"{match.group('indent')}self.{field} = "
            f'os.environ.get("{env_name}", "{default}")'
        )
        source = source[:match.start()] + new + source[match.end():]
        changes.append(f"Config.{field} -> ${env_name}")

    old_padding = "np.repeat(sequence[-1][None, :], self.feature_dim - T_new, axis=0)"
    if old_padding in source:
        source = source.replace(old_padding,
                                "np.repeat(sequence[-1][None, :], self.seq_len - T_new, axis=0)", 1)
        changes.append("repeat_last pad length")

    verbose_pattern = re.compile(r"(?m)^[ \t]*verbose=True[ \t]*(?:#[^\n]*)?\n")
    source, count = verbose_pattern.subn("", source, count=1)
    if count:
        changes.append("remove deprecated ReduceLROnPlateau.verbose")

    replacements = {
        "RUN_HYPERPARAMETER_SEARCH = False": (
            'RUN_HYPERPARAMETER_SEARCH = os.environ.get("CAG_MODE", "").lower() == "search"',
            "CAG_MODE=search",
        ),
        "REPRODUCE_SPECIFIC_RUN = True": (
            'REPRODUCE_SPECIFIC_RUN = os.environ.get("CAG_MODE", "").lower() == "reproduce"',
            "CAG_MODE=reproduce",
        ),
        'specific_session_timestamp = "0"': (
            'specific_session_timestamp = os.environ.get("CAG_SESSION", "0")',
            "CAG_SESSION",
        ),
    }
    for old, (new, description) in replacements.items():
        if old in source:
            source = source.replace(old, new, 1)
            changes.append(description)

    # Verify Python syntax before returning a proposed edit.
    compile(source, str(TARGET), "exec")
    return source, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Describe proposed changes")
    mode.add_argument("--apply", action="store_true", help="Back up and patch file")
    args = parser.parse_args()

    original = TARGET.read_text(encoding="utf-8")
    revised, changes = patch_source(original)
    if not changes:
        print("No changes needed; compatible patches already present.")
        return
    print("Proposed changes:")
    for change in changes:
        print(" -", change)
    if args.apply:
        backup = TARGET.with_suffix(TARGET.suffix + ".bak")
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite existing backup: {backup}")
        backup.write_text(original, encoding="utf-8")
        TARGET.write_text(revised, encoding="utf-8")
        print("Applied compatibility changes and saved backup at", backup)
        print("Review with: git diff -- VAE_reproducible_improved.py")
    else:
        print("Dry run only; no files modified. Use --apply to opt in.")


if __name__ == "__main__":
    main()
