# Progress

Where the project stands. The record of what changed is [CHANGELOG.md](CHANGELOG.md), including
the corrections to `v1.0.0`, and the record of what went wrong is
[docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md).

## Current state

`v1.0.0` is published. `v1.1.0` is in preparation on `v1.1-dev`: a hardening release that adds no
algorithm.

The results in `experiments/results` were measured at commit `<!-- num:BenchCommit -->a27a69ed5ec4<!-- /num -->` on an <!-- num:GpuName -->NVIDIA GeForce RTX 5070<!-- /num -->,
with session drift of <!-- num:RepeatabilitySpread -->0.17<!-- /num --> percent.

- Shared memory tiling is worth a factor of <!-- num:TilingSpeedupAtMaxN -->1.96<!-- /num --> at <!-- num:LadderMaxN -->200,000<!-- /num --> particles, taking
  the step from <!-- num:NaiveMsAtMaxN -->84.73<!-- /num --> to <!-- num:TiledMsAtMaxN -->43.16<!-- /num --> ms, at <!-- num:TiledGigaInteractionsAtMaxN -->927<!-- /num --> billion
  interactions per second.
- Kernel fusion is measurably faster up to 150,000 particles, where the paired ratio is
  <!-- num:FusedOverTiledRatioAtHundredFiftyThousand -->0.9909<!-- /num -->, and not resolved at <!-- num:LadderMaxN -->200,000<!-- /num -->, where tiled trials are
  bimodal for a reason not yet understood.
- The default tile width is <!-- num:BestTileWidth -->256<!-- /num -->.
- The fluid solver runs 50,000 particles at <!-- num:SphMsAtFiftyThousand -->0.345<!-- /num --> ms per step.
- All <!-- num:GateCount -->14<!-- /num --> physics gates pass. The interior density mean error is <!-- num:SphDensityErrorPercent -->0.03<!-- /num --> percent
  by calibration, with an RMS deviation of <!-- num:SphDensityRmsPercent -->1.49<!-- /num --> percent.

## Open questions

- Why tiled trials at <!-- num:LadderMaxN -->200,000<!-- /num --> particles are bimodal.
- What the fused kernel's gain comes from. The bytes it removes are too few to explain it.
- Why the tiling factor is larger at 10,000 particles than at any other size.
- The fluid start is better but not quiet. A hydrostatic test should own that.
- The first continuous integration run of the release build has not happened yet.
