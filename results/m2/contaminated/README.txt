Benchmark records NOT used in the M2 report.
- benchmark_overlapping_runs.jsonl: two Claude sessions ran sweeps concurrently on GPU 0 (2026-11-21); duplicated
  configurations with inconsistent timings.
- benchmark_session1_tail_possibly_overlapping.jsonl, benchmark_gpu5_session2_partial.jsonl: runs that may have overlapped
  in time with the other session's runs (CPU contention).
The M2 report uses results/m2/benchmark_clean.jsonl: one sequential sweep on a single GPU, no other runs of ours active.
