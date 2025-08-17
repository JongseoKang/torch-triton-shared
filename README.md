# triton-shared
torch2.0+triton-shared

## Installation
1. clone triton-shared
```
export TRITON_PLUGIN_DIRS=$(pwd)/triton_shared

git clone --recurse-submodules https://github.com/microsoft/triton-shared.git triton_shared
cd triton_shared/triton
python3 -m venv .venv --prompt triton
source .venv/bin/activate
```

2. install torch family
```
pip3 install ninja cmake wheel pytest pybind11 setuptools pillow numpy requests
pip3 install torch==2.7.1
pip3 install torchvision==0.22.1 --no-deps
```

3. install hf famliy
```
pip3 install transformers huggingface_hub tokenizers datasets
```

4. install custom files
```
cd ~
git clone https://github.com/JongseoKang/torch-triton-shared
cd torch-triton-shared
./install.sh
```

5. install triton-shared
```
cd triton_shared/triton
pip3 install -e . --no-build-isolation
```

6. set env var
```
export TRITON_SHARED_OPT_PATH=""
export LLVM_BINARY_BIN=""
```