"""Resource locations for source checkouts and immutable installed packages.

Source discovery never requires generated JSON. Explicit build commands write
delivery metadata under build/generated; wheels carry that metadata as data/.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def source_root():
    root = PACKAGE_ROOT.parent
    return (
        root
        if (root / "pyproject.toml").is_file() and (root / "csrc").is_dir()
        else None
    )


def generated_root(root=None):
    root = Path(root) if root is not None else source_root()
    if root is None:
        raise ValueError("generation requires a source checkout")
    return root / "build/generated"


def data_root(root=None):
    if root is not None:
        root = Path(root)
        # Explicit manifest clients can retain their independent project layout.
        legacy = root / "flashinfer/data"
        if legacy.is_dir() and not (root / "pyproject.toml").is_file():
            return legacy
        return generated_root(root) / "flashinfer/data"
    return (
        data_root(source_root()) if source_root() is not None else PACKAGE_ROOT / "data"
    )


def recipe_root(root=None):
    if root is not None:
        root = Path(root)
        if not (root / "pyproject.toml").is_file() and (root / "csrc/recipes").is_dir():
            return root / "csrc/recipes"
        return generated_root(root) / "csrc/recipes"
    return (
        recipe_root(source_root())
        if source_root() is not None
        else PACKAGE_ROOT / "data/build/csrc/recipes"
    )


def resource_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("resource must be package relative")
    root = source_root()
    if root is not None:
        authored = root / "tools/metadata" / relative
        if authored.is_file():
            return authored
    return data_root() / relative
