SRIE Project 'Caliper'

## Running larger local models on an 8GB GPU

`sycophancy_evaluation.py` loads local models according to `load_mode` (see the `eval_loop` calls in `main()`):

- `None` (default): default precision, e.g. `Qwen/Qwen3-0.6B`.
- `"4bit"`: 4-bit NF4 quantisation with bitsandbytes, e.g. `Qwen/Qwen3-8B` (peak ~6.6GB of GPU memory).
- `"4bit-offload"`: 4-bit decoder layers on the GPU, with the input embeddings and output head kept in CPU RAM,
  e.g. `Qwen/Qwen3-14B` (peak ~7.6GB of GPU memory, plus ~5GB of RAM).

Quantised runs get a `-4bit` suffix in the results file name.

On a laptop RTX 2000 Ada (8GB), the 305 `keep_core_k4` questions take about 22 minutes with Qwen3-8B and about
1 hour with Qwen3-14B. Weights are downloaded once in full precision (~16GB and ~30GB) and quantised when loaded.

### Environment

These modes need a CUDA build of PyTorch and bitsandbytes. For example, with uv on Windows:

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe transformers accelerate bitsandbytes pandas requests matplotlib scipy
```

On Linux or macOS, use `.venv/bin/python`. Tested with torch 2.11 (CUDA 12.8), transformers 5.18,
bitsandbytes 0.50 and accelerate 1.15; `"4bit-offload"` relies on accelerate internals, so check it again
after upgrading.

## Greedy decoding in earlier results

Before the commit "make temperature 0.0 actually greedy", `temperature=0.0` did not give greedy decoding for
Qwen3 models: `generate()` falls back to the model's own generation config, and Qwen3 ships with
`do_sample=True` (temperature 0.6, top-k 20, top-p 0.95). Results produced before that commit, including the
earlier Qwen3-0.6B runs, are sampled rather than greedy.
