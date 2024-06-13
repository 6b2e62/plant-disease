.PHONY: download-dataset unzip-dataset resize-dataset sobel-dataset login
.PHONY: docker-run docker-build check-gpu
.PHONY: create-mobilenet-job create-efficientnet-job create-resnet50-job
.PHONY: run-optuna run-transfer-learning

PROJECT = "Detection of plant diseases"
ENTITY = "uczenie-maszynowe-projekt"

download-dataset:
	python3 src/file_manager/file_manager.py --download --unzip

unzip-dataset:
	python3 src/file_manager/file_manager.py --unzip

resize-dataset:
	python3 src/file_manager/file_manager.py --resize --shape 128 128 --source "original_dataset"

sobel-dataset:
	python3 src/file_manager/file_manager.py --sobel --source "resized_dataset"

login:
	wandb login $$(cat "$$API_KEY_SECRET")

docker-run:
	docker compose run --entrypoint=/bin/bash gpu

docker-build:
	docker compose build

check-gpu:
	python3 ./gpu_check.py

# wandb commands
create-mobilenet-job:
	wandb job create --project $(PROJECT) --entity $(ENTITY) --name "mobilenet" git git@github.com:6b2e62/plant-disease.git --entry-point "python3 main.py --model mobilenet"

create-efficientnet-job:
	wandb job create --project $(PROJECT) --entity $(ENTITY) --name "efficientnet" git git@github.com:6b2e62/plant-disease.git --entry-point "python3 main.py --model efficientnet"

create-resnet50-job:
	wandb job create --project $(PROJECT) --entity $(ENTITY) --name "resnet50" git git@github.com:6b2e62/plant-disease.git --entry-point "python3 main.py --model resnet50"

# e. g. make run-optuna PARAMS='--model mobilenet --size 96 --trials 10 --with-checkpoints'
run-optuna:
	python3 src/optuna_trainer.py ${PARAMS}

# e. g. make run-transfer-learning PARAMS='--model mobilenet --size 96 --epochs 10 --with-checkpoints'
run-transfer-learning:
	python3 src/transfer_learning.py ${PARAMS}

# e.g. make run-frontend PARAMS='--model resnet50 --size 256 --path weights/transfer_learning/resnet50v2_256.keras'
run-frontend:
	python3 src/frontend.py ${PARAMS}