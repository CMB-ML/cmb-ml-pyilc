import sys

from pyilc.main import main


def run_ilc(cfg_path):
    original_argv = sys.argv
    try:
        sys.argv = ["pyilc", str(cfg_path)]
        main()
    finally:
        sys.argv = original_argv
