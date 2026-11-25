"""Export saved video annotations from the command line (no browser required)."""
import argparse
from pathlib import Path
from video_dataset import DEFAULT_PROJECT, DEFAULT_OUTPUT, export_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-fps", type=float, default=5)
    args = parser.parse_args()
    try:
        report = export_dataset(args.project, args.output, args.sample_fps,
                                progress=lambda fraction, name: print(f"{fraction:6.1%} {name}", flush=True))
    except Exception as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Saved {report['rows']} samples: {args.output}")
    print(f"Report: {args.output.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
