"""Print per-kernel cycle breakdown for Qwen3-A3B decode at chosen tile config."""

from __future__ import annotations

import argparse

from fllm.cycle_sim import ModelShape, TileSpec, estimate_token


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lanes", type=int, default=32)
    p.add_argument("--tile-rows", type=int, default=16)
    p.add_argument("--num-tiles", type=int, default=4)
    p.add_argument("--fmax-mhz", type=float, default=600.0)
    p.add_argument("--avg-ctx", type=int, default=1024)
    return p


def main() -> None:
    args = build_parser().parse_args()
    tile = TileSpec(
        lanes=args.lanes, tile_rows=args.tile_rows,
        num_tiles=args.num_tiles, fmax_mhz=args.fmax_mhz,
    )
    shape = ModelShape(avg_ctx=args.avg_ctx)
    r = estimate_token(shape, tile)

    print(f"tile: lanes={tile.lanes} rows={tile.tile_rows} tiles={tile.num_tiles} fmax={tile.fmax_mhz}MHz")
    print(f"model: Qwen3-A3B shape, ctx={shape.avg_ctx}")
    print()
    print("per-block breakdown (cycles):")
    for name, cyc in r.breakdown_per_block.items():
        share = cyc / r.block_cycles * 100
        print(f"  {name:12} {cyc:>12,}  {share:5.1f}%")
    print(f"  -- per-block total: {r.block_cycles:,} cycles")
    print()
    print(f"per-token total (x{shape.num_layers} layers + lm_head):")
    print(f"  block * 48 = {r.block_cycles * shape.num_layers:,}")
    print(f"  lm_head    = {r.lm_head:,}")
    print(f"  cycles/tok = {r.cycles_per_token:,}")
    compute_ms = r.cycles_per_token / (tile.fmax_mhz * 1e3)
    compute_tps = 1000.0 / compute_ms if compute_ms > 0 else 0.0
    print(f"  compute ceiling: {compute_ms:.3f} ms/tok ({compute_tps:.1f} tok/s)")
    print(f"  hbm     ceiling: {r.hbm_ms_per_token:.3f} ms/tok ({r.hbm_tokens_per_s:.1f} tok/s)")
    print(f"  hbm bytes/tok  : {r.bytes_per_token/1e6:.1f} MB")
    print()
    print(f"BOTTLENECK = {r.bottleneck.upper()}")
    print(f"  realized: {r.ms_per_token:.3f} ms/tok ({r.tokens_per_s:.1f} tok/s)")


if __name__ == "__main__":
    main()
