import os
import sys
from pathlib import Path

import pytest

from .test_full import gen_toy

this_dir = Path(__file__).parent


def test_fit(gen_toy):
    print(os.getcwd())
    sys.argv = [
        "fit.py",
        "-c",
        str(this_dir / "config_toy.yml"),
        "-i",
        str(this_dir / "exp_params.json"),
        "--no-GPU",
    ]
    exec("from fit import main; main()")


def test_fit2(gen_toy):
    print(os.getcwd())
    sys.argv = [
        "fit.py",
        "-c",
        str(this_dir / "config_toy.yml"),
        "-i",
        str(this_dir / "exp_params.json"),
        "--no-GPU",
        "--printer=normal",
    ]
    exec("from fit import main; main()")


def test_main_fit(gen_toy):
    from tf_pwa.app.fit import fit

    toyconfigpath = this_dir / "config_toy.yml"
    expparamspath = this_dir / "exp_params.json"

    fit(toyconfigpath, expparamspath)
