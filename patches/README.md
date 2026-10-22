# Patches

`d1-prefix-cache.patch` is the llama.cpp change, generated as

    git diff 06cad0b9e77315bd2930bd4f70a6d7b37b2a01c1 <branch> -- common tools/server/

against upstream base `06cad0b`. Code only: `common/` (CLI flags) and
`tools/server/` (decision prefix checkpoints + bounded trunk-ID cache).
Default OFF; production behaviour without the flags is unchanged.

The same code is maintained on the public fork branch `d1-prefix-cache`:
<https://github.com/urawazakun/llama.cpp/tree/d1-prefix-cache>

Build like any llama.cpp CUDA build (MSVC + CUDA 11.8 + sm_70 was tested;
any normal toolchain works), then serve d1-3B-GGUF
(<https://huggingface.co/LiquidAI/d1-3B-GGUF>) with e.g.:

    llama-server -m d1-3B-Q4_K_M.gguf --mmproj mmproj-d1-3B-F16.gguf \
      --decision-prefix-cache --decision-trunk-cache 512 --metrics
