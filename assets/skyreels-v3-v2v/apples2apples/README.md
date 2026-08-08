V2V CPU-offload apples-to-apples run is documented in the sibling A2V worktree at:
../../skyreels-v3-a2v-impl/results/apples2apples/BENCH_SUMMARY.md

The V2V-side blocking finding is preserved here:
  logs/h2_v2v_offload_gpu1_v3.log — CPU-offload dtype bug
    (Input type (CPUBFloat16Type) vs weight type (CUDABFloat16Type))

Same class of bug as A2V's `CLIPModel.visual` under offload; blocks a true
apples-to-apples V2V speed number vs Skywork's `--offload` mode.
