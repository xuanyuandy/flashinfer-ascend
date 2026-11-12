"""Compatibility entry; numerical semantics live in the author directory."""

import argparse
from flashinfer.operators.catalog import cases_module
from flashinfer.operators.testing import run as _run

NAME = "grouped_matmul_swiglu_quant"
sample = cases_module(NAME).sample
golden = cases_module(NAME).golden


def run(mode):
    return _run(NAME, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="matrix", choices=("matrix", "one", "missing", "corrupt", "installed"))
    run(parser.parse_args().mode)
