#!/usr/bin/env python3
"""
check-doc-drift.py
------------------
Detects which documentation files need updating after code changes.

Compares current file hashes against docs/meta/SCAN_MANIFEST.json and
outputs a prioritized list of documentation update tasks for a coding agent.

Usage:
    python scripts/check-doc-drift.py                    # check all tracked files
    python scripts/check-doc-drift.py --since HEAD~1     # check files changed in last commit
    python scripts/check-doc-drift.py --since <hash>     # check files changed since a commit
    python scripts/check-doc-drift.py --file path/to/file.py  # check one specific file

Output:
    Prints a structured update report to stdout.
    Exits 0 if no drift detected, 1 if updates are needed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path("docs/meta/SCAN_MANIFEST.json")
UPDATE_LOG_PATH = Path("docs/meta/UPDATE_LOG.md")

# Files in these directories are never tracked for drift
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "execution_results",
    "wandb", "mlruns", ".venv", "env", "dist-packages", ".cache",
}

# Files with these extensions are never tracked
SKIP_EXTENSIONS = {
    ".csv", ".tsv", ".parquet", ".arrow",
    ".pt", ".pth", ".pkl", ".ckpt", ".safetensors", ".onnx",
    ".npy", ".npz", ".bin", ".h5", ".hdf5",
    ".pyc", ".pyo", ".lock",
}

# Maps doc directory prefixes to the skills needed to update them
DOC_SKILL_MAP = {
    "docs/ai_context/modules/": ".codex/skills/update-codebase-map/SKILL.md",
    "docs/ai_context/SYSTEM_STATE.md": ".codex/skills/update-docs/SKILL.md",
    "docs/ai_context/INDEX.md": ".codex/skills/update-codebase-map/SKILL.md",
    "docs/human/modules/": ".codex/skills/write-human-doc/SKILL.md",
    "docs/human/ARCHITECTURE.md": ".codex/skills/write-human-doc/SKILL.md",
    "docs/human/IMPLEMENTATION_STATUS.md": ".codex/skills/write-human-doc/SKILL.md",
    "docs/codebase_map/": ".codex/skills/update-codebase-map/SKILL.md",
}


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return "unreadable"


def load_manifest() -> dict:
    """Load the scan manifest. Exits with a clear error if it does not exist."""
    if not MANIFEST_PATH.exists():
        print(
            f"ERROR: {MANIFEST_PATH} not found.\n"
            "Run the initial documentation scan first:\n"
            "  Use task: .codex/tasks/document-codebase.md",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(MANIFEST_PATH) as f:
        return json.load(f)


def get_git_changed_files(since: str) -> list[str]:
    """
    Return list of files changed since a git ref.
    Returns empty list if git is not available or ref is invalid.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since, "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.CalledProcessError as e:
        print(f"Warning: git diff failed ({e}). Falling back to full hash check.", file=sys.stderr)
        return []


def should_skip(path: Path) -> bool:
    """Return True if this file should never be tracked."""
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    if path.suffix in SKIP_EXTENSIONS:
        return True
    return False


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def find_drifted_files(manifest: dict, filter_paths: list[str] | None = None) -> list[dict]:
    """
    Compare current file hashes against manifest.
    Returns list of drifted file entries with current hash and change type.
    """
    drifted = []
    manifest_by_path = {entry["path"]: entry for entry in manifest.get("files_scanned", [])}

    if filter_paths is not None:
        # Only check the specified paths
        paths_to_check = {p: manifest_by_path.get(p) for p in filter_paths}
    else:
        paths_to_check = manifest_by_path

    for rel_path, manifest_entry in paths_to_check.items():
        file_path = Path(rel_path)

        if should_skip(file_path):
            continue

        if not file_path.exists():
            drifted.append({
                **manifest_entry,
                "change_type": "DELETED",
                "current_hash": None,
            })
            continue

        current_hash = sha256_file(file_path)
        if manifest_entry is None:
            drifted.append({
                "path": rel_path,
                "type": "unknown",
                "change_type": "NEW_UNTRACKED",
                "current_hash": current_hash,
                "documentation_targets": [],
            })
        elif current_hash != manifest_entry.get("sha256"):
            drifted.append({
                **manifest_entry,
                "change_type": "MODIFIED",
                "current_hash": current_hash,
            })

    return drifted


def find_untracked_new_files(manifest: dict) -> list[str]:
    """Find .py and .md files that exist now but were not in the original scan."""
    manifest_paths = {entry["path"] for entry in manifest.get("files_scanned", [])}
    project_root = Path(manifest.get("project_root", "."))
    untracked = []

    for ext in ("*.py", "*.md", "*.yaml", "*.toml"):
        for f in project_root.rglob(ext):
            if should_skip(f):
                continue
            rel = str(f.relative_to(project_root))
            if rel not in manifest_paths:
                untracked.append(rel)

    return sorted(untracked)


# ---------------------------------------------------------------------------
# Impact mapping
# ---------------------------------------------------------------------------

def map_to_doc_targets(drifted_files: list[dict], manifest: dict) -> dict[str, list[str]]:
    """
    Map drifted source files to the documentation files that cover them.
    Returns: { doc_path: [source_file_paths_that_affect_it] }
    """
    doc_to_sources: dict[str, list[str]] = {}

    for entry in drifted_files:
        targets = entry.get("documentation_targets", [])
        if not targets:
            # Infer targets from path if not recorded in manifest
            targets = infer_doc_targets(entry["path"], manifest)

        for doc_path in targets:
            doc_to_sources.setdefault(doc_path, []).append(entry["path"])

    return doc_to_sources


def infer_doc_targets(source_path: str, manifest: dict) -> list[str]:
    """
    Infer which documentation files should be updated for a given source file.
    Used when documentation_targets is not recorded in the manifest.
    """
    targets = []
    path = Path(source_path)

    # Find which module this file belongs to
    module = infer_module(path, manifest)

    if module:
        targets += [
            f"docs/ai_context/modules/{module}.md",
            f"docs/human/modules/{module}.md",
            f"docs/codebase_map/modules/{module}/MODULE.md",
            f"docs/codebase_map/modules/{module}/files/{path.stem}.md",
        ]
        targets.append("docs/ai_context/INDEX.md")

    # Any change might affect system state
    targets.append("docs/ai_context/SYSTEM_STATE.md")

    # If it's a runner or interface file, it also affects the project map
    if path.name in ("runners.py", "interfaces.py", "api.py"):
        targets.append("docs/codebase_map/PROJECT_MAP.md")

    return list(dict.fromkeys(targets))  # deduplicate, preserve order


def infer_module(path: Path, manifest: dict) -> str | None:
    """Infer the module name from a file path using the manifest's module registry."""
    modules = {m["name"]: m["directory"].rstrip("/") for m in manifest.get("modules_found", [])}
    path_str = str(path)

    for module_name, module_dir in modules.items():
        if path_str.startswith(module_dir + "/") or path_str.startswith(module_dir):
            return module_name

    # Fallback: use top-level directory name
    parts = path.parts
    if len(parts) > 1:
        return parts[0]

    return None


def get_skill_for_doc(doc_path: str) -> str:
    """Return the skill to use when updating a given documentation file."""
    for prefix, skill in DOC_SKILL_MAP.items():
        if doc_path.startswith(prefix) or doc_path == prefix:
            return skill
    return ".codex/skills/update-docs/SKILL.md"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_agent_prompt(
    drifted: list[dict],
    doc_to_sources: dict[str, list[str]],
    untracked: list[str],
    manifest: dict,
) -> str:
    """Build a structured prompt for a coding agent to execute the documentation updates."""

    lines = [
        "# Documentation Update Required",
        f"> Generated: {datetime.now(timezone.utc).isoformat()}",
        f"> Project: {manifest.get('project_name', 'unknown')}",
        "",
    ]

    if not drifted and not untracked:
        lines += ["## No drift detected. All documentation is current.", ""]
        return "\n".join(lines)

    # Summary
    lines += [
        "## Summary",
        f"- Source files changed: {len(drifted)}",
        f"- New untracked files: {len(untracked)}",
        f"- Documentation files to update: {len(doc_to_sources)}",
        "",
    ]

    # Changed source files
    if drifted:
        lines += ["## Changed Source Files", ""]
        for entry in drifted:
            change = entry.get("change_type", "MODIFIED")
            lines.append(f"- `{entry['path']}` — **{change}**")
        lines.append("")

    # New untracked files
    if untracked:
        lines += [
            "## New Files (Not in Scan Manifest)",
            "These files were added after the initial scan. They need to be added to the manifest",
            "and documented. For each, create the appropriate codebase_map entry.",
            "",
        ]
        for f in untracked:
            lines.append(f"- `{f}`")
        lines.append("")

    # Ordered update tasks
    lines += [
        "## Update Tasks",
        "Execute in this order. Read the referenced skill before each task.",
        "",
    ]

    # Sort: system state first (most broadly affected), then module docs, then codebase map
    priority_order = [
        "docs/ai_context/SYSTEM_STATE.md",
        "docs/ai_context/INDEX.md",
        "docs/codebase_map/PROJECT_MAP.md",
    ]
    sorted_docs = sorted(
        doc_to_sources.keys(),
        key=lambda d: (
            0 if d in priority_order else
            1 if d.startswith("docs/ai_context/") else
            2 if d.startswith("docs/codebase_map/") else
            3
        )
    )

    for i, doc_path in enumerate(sorted_docs, 1):
        sources = doc_to_sources[doc_path]
        skill = get_skill_for_doc(doc_path)
        lines += [
            f"### Task {i}: Update `{doc_path}`",
            f"**Skill:** `{skill}`",
            f"**Because these files changed:**",
        ]
        for s in sources:
            lines.append(f"  - `{s}`")
        lines += [
            f"**Instructions:** Read `{skill}` then update `{doc_path}` to reflect",
            f"the current state of the changed files listed above.",
            "",
        ]

    # Manifest update instruction
    lines += [
        "## Final Step: Update Scan Manifest",
        "After completing all documentation updates, update `docs/meta/SCAN_MANIFEST.json`:",
        "- Update `sha256` for every modified file",
        "- Add entries for any new untracked files",
        "- Update `documentation_targets` for new files",
        "- Append an entry to `docs/meta/UPDATE_LOG.md`",
        "",
        "### UPDATE_LOG.md entry format:",
        "```markdown",
        f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "**Trigger:** <what changed>",
        "**Files changed:** <list>",
        "**Docs updated:** <list>",
        "**Agent model:** <model>",
        "```",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect documentation drift and generate update tasks."
    )
    parser.add_argument(
        "--since",
        metavar="GIT_REF",
        help="Only check files changed since this git ref (e.g., HEAD~1, abc1234)",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Only check this specific file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the agent prompt, no status messages",
    )
    args = parser.parse_args()

    manifest = load_manifest()

    # Determine which files to check
    filter_paths = None

    if args.file:
        filter_paths = [args.file]
    elif args.since:
        changed = get_git_changed_files(args.since)
        if changed:
            filter_paths = changed
        # If git failed, fall back to full check

    # Find drift
    drifted = find_drifted_files(manifest, filter_paths)
    untracked = find_untracked_new_files(manifest) if filter_paths is None else []

    # Map to doc targets
    doc_to_sources = map_to_doc_targets(drifted, manifest)

    # Build and print agent prompt
    prompt = build_agent_prompt(drifted, doc_to_sources, untracked, manifest)
    print(prompt)

    # Exit code: 1 if updates needed, 0 if clean
    needs_update = bool(drifted or untracked)
    if not args.quiet:
        if needs_update:
            print(f"\n[check-doc-drift] Updates needed for {len(doc_to_sources)} documentation files.", file=sys.stderr)
        else:
            print("[check-doc-drift] Documentation is current. No updates needed.", file=sys.stderr)

    sys.exit(1 if needs_update else 0)


if __name__ == "__main__":
    main()