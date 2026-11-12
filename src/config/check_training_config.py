"""Print a resolved training recipe without importing torch or touching datasets."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from config.training_args import build_parser
from config.training_config import training_summary


def main(argv=None):
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    try:
        print(json.dumps(training_summary(args), indent=2))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
