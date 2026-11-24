from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "lint_no_fabrication", ROOT / "tools" / "lint_no_fabrication.py"
)
assert spec is not None
assert spec.loader is not None
lint = importlib.util.module_from_spec(spec)
sys.modules["lint_no_fabrication"] = lint
spec.loader.exec_module(lint)


@pytest.mark.parametrize(
    "source",
    [
        "import random\n",
        "from random import random\n",
        "import numpy as np\nx = np.random.rand(3)\n",
        "import numpy as np\nrng = np.random.default_rng(0)\n",
        "import numpy\nx = numpy.random.normal(0, 1)\n",
        "from numpy.random import default_rng\n",
        "import torch\nx = torch.randn(3)\n",
        "import torch\nx = torch.rand_like(y)\n",
    ],
)
def test_random_sources_are_rejected_in_product_code(source: str) -> None:
    assert lint.check_source(source, "models/adapters/anything.py")


def test_seeded_generator_is_allowed_only_in_allow_listed_modules() -> None:
    seeded = "import numpy as np\nrng = np.random.default_rng(1234)\n"
    assert lint.check_source(seeded, "eval/bootstrap.py") == []
    assert lint.check_source(seeded, "train/augment.py") == []
    assert lint.check_source(seeded, "pipeline.py") != []


def test_unseeded_and_legacy_rng_are_rejected_even_where_seeding_is_allowed() -> None:
    assert lint.check_source(
        "import numpy as np\nrng = np.random.default_rng()\n", "eval/bootstrap.py"
    )
    assert lint.check_source("import numpy as np\nnp.random.seed(0)\n", "eval/bootstrap.py")
    assert lint.check_source("import random\n", "train/augment.py")


def test_ordinary_code_passes() -> None:
    source = "import numpy as np\nimport secrets\nimport uuid\nx = np.zeros(3).mean()\nt = secrets.token_urlsafe(32)\n"
    assert lint.check_source(source, "api/auth.py") == []


def test_the_real_package_is_clean() -> None:
    assert lint.check_tree(ROOT / "src" / "medai") == []
