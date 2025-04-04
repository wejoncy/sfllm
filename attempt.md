1. CUDA GRAPH
2. Customized model definition/Model loader
3. KV cache managerment
4. over-lapped scheduling/ Overlap CPU/GPU
5. ragged batch/ P&D& request
6. ragged attention with torch naive implementation
7. replaced rms kernel
8. triton kernel

However, When I dig into those optmization, all things are blocked by the item 2, I need a customized Model definition and a Model loader, This cost too much time to debug.