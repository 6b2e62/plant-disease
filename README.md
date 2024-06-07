# Harmonogram

| Data                        | Działanie
|----------------------------:|:------------------------------------------------------------|
| 19.04.2024                  | Prezentacja problemu i zbioru danych
|                             | Prezentacja technologii wykorzystywanych w projekcie
|                             | Wstępny szkic MLOps pipeline
|                             | Szkic aplikacji (backend + frontend)
| 08.06.2024                  | Prezentacja MLOps pipeline
|                             | Prezentacja aplikacji
|                             | Prezentacja wyników eksperymentów
| 15.06.2024                  | Prezentacja działania systemu
|                             | Prezentacja wyników i skuteczności wybranego modelu

# Dokumentacja

[Link do dokumentacji](https://uam-my.sharepoint.com/personal/krzboj_st_amu_edu_pl/_layouts/15/doc.aspx?sourcedoc={dc695bbe-68d1-4947-8c29-1d008f252a3b}&action=edit)

# Setup

1. Install Docker on your local system.
2. To build docker image and run the shell, use Makefile
```bash
make docker-build
make docker-run
```
3. Get your API key from https://wandb.ai/settings#api, and add the key to **secrets.txt** file.
4. After running the container
```bash
make login # to login to WanDB.
make check-gpu # to verify if GPU works
```
5. If needed, to manually run containers, run:
```bash
docker build -t gpu api_key="<wandb_api_key>" .
docker run --rm -it --gpus all --entrypoint /bin/bash gpu
```

# Local WSL CUDA

It might be required to export environment variables after CUDA toolkit installation. Check `Dockerfile` as an example.

```bash
export CUDNN_PATH="/home/username/.local/lib/python3.10/site-packages/nvidia/cudnn/"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda-12.2/lib64"
export PATH="$PATH":"/usr/local/cuda-12.2/bin"
```

# Training models
```bash
python3 optuna_trainer.py --model mobilenet --with-checkpoints --size 96 

python3 transfer_learning.py --model mobilenet --with-checkpoints --size 96
```