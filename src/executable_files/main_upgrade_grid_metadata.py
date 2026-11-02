"""Explicitly add a verified operating grid to existing metadata."""

import argparse
from pathlib import Path

from src.modules.data_managment.map_grid import MapGrid
from src.modules.data_managment.simulation_metadata import (
    upgrade_grid_metadata,
)


class GridMigrationArguments(argparse.Namespace):
    csv: list[Path]
    shaft_speed_norm: list[float]
    q_current_density: list[float]
    d_current_density: list[float]
    map_size: list[int]
    write: bool


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--shaft-speed-norm",
        type=float,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "--q-current-density",
        type=float,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "--d-current-density",
        type=float,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "--map-size",
        type=int,
        nargs=3,
        required=True,
    )
    parser.add_argument(
        "--write",
        action="store_true",
    )

    args = GridMigrationArguments()
    parser.parse_args(namespace=args)

    grid = MapGrid(
        shaft_speed_norm=(
            args.shaft_speed_norm[0],
            args.shaft_speed_norm[1],
        ),
        q_current_density=(
            args.q_current_density[0],
            args.q_current_density[1],
        ),
        d_current_density=(
            args.d_current_density[0],
            args.d_current_density[1],
        ),
        map_size=(
            args.map_size[0],
            args.map_size[1],
            args.map_size[2],
        ),
    )

    # Validate all selected sources before starting any writes.
    for path in args.csv:
        upgrade_grid_metadata(
            csv_path=path,
            grid=grid,
        )

    if not args.write:
        for path in args.csv:
            print(f"Verified (dry run): {path}")
        return

    for path in args.csv:
        upgrade_grid_metadata(
            csv_path=path,
            grid=grid,
            write=True,
        )
        print(f"Updated: {path}")


if __name__ == "__main__":
    main()
