.PHONY: download-dataset resize-dataset sobel-dataset

# Use inside docker container
download-dataset:
	python3 ./file_manager/data_manager.py --download

resize-dataset:
	python3 ./file_manager/data_manager.py --resize --shape 64 64 --source "original_dataset"

sobel-dataset:
	python3 ./file_manager/data_manager.py --sobel --source "resized_dataset"

login:
	wandb login $$(cat "$$API_KEY_SECRET")

# Outside docker
docker-run:
	docker compose run --entrypoint=/bin/bash gpu

docker-build:
	docker compose build

check-gpu:
	python3 ./gpu_check.py
