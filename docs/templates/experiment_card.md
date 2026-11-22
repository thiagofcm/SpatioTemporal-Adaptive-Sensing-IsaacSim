# <ID>

**Status:** planned
<!-- planned | training | evaluating | done | superseded by <ID>. Keep the registry row in docs/EXPERIMENTS.md in sync. -->

**Question:** <one sentence>

**Hypothesis:** <expected result, and what would change your mind>

## Setup

| | |
|---|---|
| Created | <date> |
| Git commit at creation | `<commit>` |
| Config | [<line>/configs/<ID>.yaml](../../configs/<ID>.yaml) |
| Training runs | `<line>/runs/<ID>/` |
| Evaluations | `<line>/eval/<ID>/<protocol>/<policy>/` |
| Differs from previous experiment | <the one or two things that changed> |
| Reference equivalent | <e.g. ST04 in ~/adaptive_sensor_policy, or none> |

## Commands

```bash
bin/isaacrun -g <gpu> -- python <line>/scripts/train.py --headless --config <line>/configs/<ID>.yaml
bin/isaacrun -g <gpu> -- python <line>/scripts/evaluate.py --headless --checkpoint <line>/runs/<ID>/<run>/model_<k>.pt
```

## Training runs

| Run folder | Iterations | Seed | Throughput | Notes |
|---|---|---|---|---|

## Evaluations

| Protocol | Policy | Episodes | Success | Notes |
|---|---|---|---|---|

## Results

## Conclusion and next step
