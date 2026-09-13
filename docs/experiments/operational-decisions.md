# Decisions before execution

2026-09-13: Restore the two official 0.8B training arms after the user's
explicit correction. Use 334 provisional steps, the same 2e17 cost target,
8192 output cap and original deepscaler reward as 2B/4B. Reuse immutable
source 01539b81; do not restart the four existing runs. Keep 9B excluded.

Enable the documented SGLang request-dump endpoint on active training
engines, with a separate persistent directory per run and threshold 12.
Preserve enablement timestamps and missing earlier coverage. This is logging,
not a training-reward or algorithm change. Both decisions are recorded in
the parent experiment protocol before job/endpoint mutations.
