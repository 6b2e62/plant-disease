.PHONY: download-dataset resize-dataset sobel-dataset login
.PHONY: docker-run docker-build check-gpu
.PHONY: create-mobilenet-job create-efficientnet-job create-resnet50-job
.PHONY: create-mobilenet-sweep create-efficientnet-sweep create-resnet50-sweep

PROJECT = "Detection of plant diseases"
ENTITY = "uczenie-maszynowe-projekt"

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

# wandb commands
create-mobilenet-job:
	wandb job create --project $(PROJECT) --entity $(ENTITY) --name "mobilenet" git https://github.com/6b2e62/plant-disease --entry-point "python3 main.py --model mobilenet"

create-efficientnet-job:
	wandb job create --project $(PROJECT) --entity $(ENTITY) --name "efficientnet" git https://github.com/6b2e62/plant-disease --entry-point "python3 main.py --model efficientnet"

create-resnet50-job:
	wandb job create --project $(PROJECT) --entity $(ENTITY) --name "resnet50" git https://github.com/6b2e62/plant-disease --entry-point "python3 main.py --model resnet50"

create-mobilenet-sweep:
	wandb launch-sweep ./wandb_configs/launch-sweep-mobilenet-config.yaml --queue sweeps-mobilenet

create-efficientnet-sweep:
	wandb launch-sweep ./wandb_configs/launch-sweep-efficientnet-config.yaml --queue sweeps-efficientnet

create-resnet50-sweep:
	wandb launch-sweep ./wandb_configs/launch-sweep-resnet50-config.yaml --queue sweeps-resnet50