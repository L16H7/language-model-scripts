# Running LLAMA CPP on Cloud instances

### Machine dependencies
`apt-get install -y cmake libcurl4-openssl-dev`

### Build for cuda
```
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

### RUN LLAMA CPP
`./llama-server --model /path/to/gguf --n-gpu-layers 30`

### Ngrok
```
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok
```

`ngrok config add-authtoken <token>`
`ngrok http --domain=<your_static_domain> <your_local_port>`
