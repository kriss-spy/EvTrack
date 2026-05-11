"""
Colab compatibility patches for SDSTrack upstream code.
Run this after cloning the upstream repository in Colab.

Issues patched:
1. torch._six removed in PyTorch 2.x
2. collections.Mapping / collections.Sequence removed in Python 3.10+
"""

import os
import re

UPSTREAM_DIR = "/content/sdstrack"
LOADER_PATH = os.path.join(UPSTREAM_DIR, "lib", "train", "data", "loader.py")


def patch_loader():
    """Patch lib/train/data/loader.py for PyTorch 2.x + Python 3.10+ compatibility."""
    if not os.path.exists(LOADER_PATH):
        print(f"WARNING: {LOADER_PATH} not found. Skipping patch.")
        return

    with open(LOADER_PATH, "r") as f:
        content = f.read()

    # Patch 1: Add collections.abc import if missing
    if "import collections.abc" not in content and "import collections" in content:
        content = content.replace(
            "import collections", "import collections\nimport collections.abc"
        )
        print("Added collections.abc import")

    # Patch 2: Replace torch._six imports with compatibility wrapper
    if "from torch._six import string_classes" in content:
        content = content.replace(
            "from torch._six import string_classes",
            """try:
    from torch._six import string_classes
except ImportError:
    string_classes = (str, bytes)""",
        )
        print("Patched string_classes import")

    # Patch 3: Replace collections.Mapping with collections.abc.Mapping
    content = content.replace("collections.Mapping", "collections.abc.Mapping")
    print("Patched collections.Mapping")

    # Patch 4: Replace collections.Sequence with collections.abc.Sequence
    content = content.replace("collections.Sequence", "collections.abc.Sequence")
    print("Patched collections.Sequence")

    with open(LOADER_PATH, "w") as f:
        f.write(content)

    print(f"Successfully patched {LOADER_PATH}")


def patch_int_classes():
    """Patch int_classes fallback if torch._six import fails."""
    if not os.path.exists(LOADER_PATH):
        return

    with open(LOADER_PATH, "r") as f:
        content = f.read()

    # Make sure int_classes fallback is safe
    if "from torch._six import int_classes" in content:
        content = content.replace(
            "from torch._six import int_classes",
            """try:
        from torch._six import int_classes
    except ImportError:
        int_classes = int""",
        )
        with open(LOADER_PATH, "w") as f:
            f.write(content)
        print("Patched int_classes import")


def scan_for_other_issues():
    """Scan for other potential compatibility issues."""
    issues = []

    # Check for other torch._six usages
    for root, dirs, files in os.walk(UPSTREAM_DIR):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r") as file:
                    content = file.read()
                if "torch._six" in content and "try:" not in content:
                    issues.append(f"Potential torch._six issue in {path}")

    if issues:
        print("\nOther potential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo other obvious compatibility issues found.")


if __name__ == "__main__":
    print("Applying SDSTrack Colab compatibility patches...")
    patch_loader()
    patch_int_classes()
    scan_for_other_issues()
    print("\nDone. You can now run: python tracking/create_default_local_file.py")
