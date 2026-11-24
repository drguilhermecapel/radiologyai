#!/usr/bin/env python3
"""Static guard for the golden rule: product code must not manufacture numbers.

Fails when ``src/medai`` uses a random number source outside the few modules
where seeded randomness is legitimate (training augmentation, bootstrap
resampling, patient-level splits, probe fitting). Even there, only a seeded
``numpy.random.default_rng(seed)`` / ``torch.Generator`` is accepted.

Usage: python tools/lint_no_fabrication.py [path ...]   (default: src/medai)
"""

from __future__ import annotations

import ast
import fnmatch
import sys
from pathlib import Path

# Files (relative to the package root) where SEEDED randomness is allowed.
SEEDED_RNG_ALLOWED = (
    "eval/bootstrap.py",
    "eval/resampling.py",
    "data/splits.py",
    "train/*.py",
    "probes/fit.py",
)

FORBIDDEN_MODULES = {"random"}
LEGACY_NUMPY = {
    "rand", "randn", "randint", "random", "random_sample", "ranf", "sample", "choice", "normal",
    "uniform", "seed", "shuffle", "permutation", "standard_normal", "beta", "binomial", "poisson",
    "RandomState",
}  # fmt: skip
TORCH_RANDOM = {
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like", "normal",
    "bernoulli", "multinomial", "poisson", "manual_seed",
}  # fmt: skip


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _allowed(relative: str) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in SEEDED_RNG_ALLOWED)


def check_source(source: str, relative: str) -> list[tuple[int, str]]:
    """Return ``(line, message)`` violations for one file."""
    tree = ast.parse(source, filename=relative)
    allowed = _allowed(relative)
    problems: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    problems.append(
                        (node.lineno, f"import of '{alias.name}' (stdlib random is never allowed)")
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in FORBIDDEN_MODULES:
                problems.append(
                    (node.lineno, f"import from '{module}' (stdlib random is never allowed)")
                )
            if module in ("numpy.random", "torch.random") and not allowed:
                problems.append(
                    (node.lineno, f"import from '{module}' outside the seeded-RNG allow-list")
                )
        elif isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted is None:
                continue
            head, _, tail = dotted.partition(".")
            if head in ("np", "numpy") and tail.startswith("random"):
                name = tail.split(".")[-1]
                if name in LEGACY_NUMPY and tail != "random":
                    problems.append(
                        (
                            node.lineno,
                            f"legacy global RNG '{dotted}' (use a seeded default_rng in an allowed module)",
                        )
                    )
                elif not allowed and tail != "random":
                    problems.append((node.lineno, f"'{dotted}' outside the seeded-RNG allow-list"))
            elif head == "torch" and tail in TORCH_RANDOM and not allowed:
                problems.append((node.lineno, f"'{dotted}' outside the seeded-RNG allow-list"))
        elif isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted and dotted.endswith("default_rng") and not node.args and not node.keywords:
                problems.append((node.lineno, "default_rng() without an explicit seed"))
    return sorted(set(problems))


def check_tree(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for line, message in check_source(path.read_text(encoding="utf-8"), relative):
            errors.append(f"{path}:{line}: {message}")
    return errors


def main(argv: list[str]) -> int:
    roots = [Path(arg) for arg in argv] or [
        Path(__file__).resolve().parent.parent / "src" / "medai"
    ]
    errors = [error for root in roots for error in check_tree(root)]
    for error in errors:
        print(error)
    if errors:
        print(f"\n{len(errors)} violation(s) of the no-fabrication rule.")
        return 1
    print("no-fabrication lint: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
