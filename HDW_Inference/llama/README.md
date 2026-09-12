# llama.cpp inference

This folder uses the system `llama-server` package to serve the local Qwen3.8 GSQ-RCO GGUF model.

## Start

The project launcher keeps this service resident even when the WebUI remains
in online mode:

```bash
bash /home/xthd/桌面/HyperDriveWave/Scripts/start.sh
```

It is managed as the user service `hyperdrivewave-llama.service`, so a
terminal closing does not stop the local model. The WebUI keeps port `3000`;
`1919` is only the local inference backend used after switching to offline
mode.

To inspect it:

```bash
systemctl --user status hyperdrivewave-llama.service --no-pager
journalctl --user -u hyperdrivewave-llama.service -n 120 --no-pager
```

For direct debugging only:

```bash
bash /home/xthd/桌面/HyperDriveWave/HDW_Inference/llama/start.sh
```

## Defaults

- model: `HDW_Engines/LLM_Models/Qwen3.8-27B-GSQ/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- host: `0.0.0.0`
- port: `1919`
- device: `Vulkan0`
- context: `262144`

## Override

```bash
HDW_LLAMA_CTX=131072 HDW_LLAMA_DEVICE=Vulkan0 bash start.sh
```

## Notes

- The packaged `llama-server` exposes OpenAI-compatible `/v1` endpoints.
- This model file includes the `-mtp` head, but the packaged server here does not expose a separate dedicated MTP switch, so the launcher keeps the setup minimal and uses the GGUF metadata/template directly.
- `start.sh` auto-falls back to CPU `--gpu-layers 0` when the GPU is already crowded; set `HDW_LLAMA_GPU_LAYERS=all` after freeing VRAM.
- When `--gpu-layers 0` is selected, the launcher switches to `--device none` so the server stays off Vulkan entirely.
