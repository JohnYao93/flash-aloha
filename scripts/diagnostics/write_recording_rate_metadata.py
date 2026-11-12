#!/usr/bin/env python3
"""Summarize source recording rates and write metadata for a LeRobot export."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from data_processing.recording_rate_metadata import write_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument(
        'recordings', nargs='+', type=Path,
        help='Exact source episodes used by the export; shell globs are supported',
    )
    args = parser.parse_args()
    write_metadata(args.dataset_root, args.recordings)


if __name__ == '__main__':
    main()
