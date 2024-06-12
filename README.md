# Projekt

Wykrywanie chorób roślin na podstawie zdjęć liści.

Projekt był realizowany na zajęcia z inżynierii uczenia maszynowego na Uniwersytecie Adama Mickiewicza - Wydział Matematyki i Informatyki.

## Źródło danych

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset 

## Dokumentacja systemu

Projekt jest podzielony na kilka modułów
- Moduł `dataset` jest odpowiedzialny za ładowanie danych
- Moduł `file_manager` jest odpowiedzialny za pobieranie i augmentację dataset'u
- Moduł `model` zawiera implementację wszystkich testowanych modeli (ResNet50V2, MobileNetV2, EfficientNetV2B0)
- Moduł `frontend` jest to interaktywna aplikacja oparta o Gradio

## Wybór modeli

Modele zostały wybrane ze względu na liczbę parametrów, co bezpośrednio przekłada się na długość czasu uczenia.
Dane w tabeli pochodzą z [dokumentacji keras](https://keras.io/api/applications/).

| Model            | Liczba parametrów |
|------------------|-------------------|
| ResNet50V2       | 25.6M             |
| EfficientNetV2B0 | 7.2M              |
| MobileNet        | 4.3M              |

## Machine Learning pipeline

![alt text](image.png)

Do realizacji każdego z elementów skorzystaliśmy z następujących technologii
- WanDB - do śledzenia eksperymentów i wizualizacji predykcji
- Optune - hiperparametryzacja
- Tensorflow - implementacja modeli, ładowanie datasetów
- Implementacja od zera - pobieranie danych, gradient, grayscale, resize

## Augmentacja danych

Dane były augmentowane z użyciem następujących technik:
- Rotacja
- Flip horyzontalny​
- Flip wertykalny​
- Zoom​

## Przeprowadzone eksperymenty

| Eksperyment                                      | Testowane opcje                          | Wyniki                                                                                                                                            |
|--------------------------------------------------|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Transfer learning na danych o różnych rozmiarach | 64x64, 96x96, 128x128, 224x224, 256x256  | Na podstawie każdej opcji można było osiągnąć wysoką skuteczność na zbiorze testowym (>90%). Jednakże 256x256 osiągnął najlepszy rezultat, również heatmap'a była bardziej precyzyjna.       |
| Transfer learning z różnymi filtrami             | grayscale, gradient                      | Filtry nie mają pozytywnego wpływu na działanie modeli i proces uczenia.                                                                                           |
| Sposób uczenia                                   | Transfer learning, training from scratch | W obu przypadkach uzyskujemy bardzo wysokie accuracy (>90%), jednakże w przypadku transfer learning'u możemy zaobserwować, że heatmap'y skupiają się na bardziej istotnych cecach. |

## Najlepsze uzyskane wyniki z transfer learningu

| Model            | batch/accuracy | epoch/val_accuracy | batch/loss | epoch/val_loss |
|------------------|----------------|--------------------|------------|----------------|
| MobileNetV2      | 0.999          | 0.9517             | 0.00359    | 0.1877         |
| ResNet50V2       | 0.9828         | 0.9527             | 0.1169     | 0.5601         |
| EfficientNetV2B0 | 0.9827         | 0.9888             | 0.0496     | 0.0348         |

## Finalne wyniki

- Model działa z doskonałą skutecznością >= 98% na danych testowych. Zarówno w wykrywaniu gatunku rośliny jak i choroby.
- W przypadku zdjęc z internetu lub z telefonu model dobrze radzi sobie z rozpoznawaniem roślin​. Jednakże często wystawia błędną diagnozę​ choroby.
- Przez analizę heatmap'y możemy zaobserwować, że 
    - Model jest bardzo podatny na zmianę kolorów (np. ciemny lub jasny liść)​
    - Model Jest bardzo podatny na wyróżniające się kolory na zdjęciu​
    - Jest podatny na elementy poza kadrem

## Dalsze kroki - jak naprawić istniejące problemy?

Głównym problemem jest zbiór danych, który jest bardzo "laboratoryjny" przez co nie reprezentuje dobrze zdjęć wykonywanych "z ręki". Aby rozwiązać problem z danymi potrzebowalibyśmy:
- Fotografie zdjęć pod różnymi kątami​
- Fotografie o różnych porach dnia​
- Fotografie z różnymi warunkami pogodowymi (słońce, chmury)​
- Fotografie przed i po podlewaniu​
- Fotografie z różnorodnym tłem
- Fotografie w cieniu / na słońcu

W trakcie augmentacji danych, dokonujemy rotacji o pewien kąt, co zostawia czasem czarne rogi na obrazkach. Moglibyśmy ten problem rozwiązać używając np. Segment Anything do wycięcia tła i podmienić je na wiele różnorodnych opcji.

## Napotkane problemy

### WanDB

- Bardzo słaba dokumentacja
- WanDB oferują opcję kolejkowania zadań, które mogą być wykonywane przez wielu agentów jednocześnie. Niestety wymagane są uprawnienia administratora oraz Docker, dlatego nie mogliśmy tego wykorzystać na Google Colab ani na maszynach uczelnianych.
- WanDB oferuję opcję hiperparametryzacji modeli (Sweeps). Moduł sweeps Łączy się z modułem opisanym powyżej, co sprawia, że też nie mogliśmy tego użyć.
- Model Registry​ - bardzo niewygodny w obsłudze, łatwiej jest przechowywać modele w chmurze albo w repozytorium.
- Nieczytelny UI

### Problemy z danymi

- Użyte zostały bardzo konkretne gatunki poszczególnych roślin - np. liście jabłka są bardzo ciemne​
- Zdjęcia są stosunkowo niskiej jakości​
- Zdjęcia nie odwzorowują dobrze roślin na żywo​
- Każde zdjęcie zostało wykonane pod +- tym samym kątem

### Inne problemy

#### Zasoby sprzętowe​

- Nvidia (3070, 4050M, 3060TI) 6-8GB GPU RAM wystarcza na transfer learning wybranych modeli oraz na uczenie MobileNetV2 od zera​
- AMD (Radeon 7800XT) nie nadaje się do uczenia maszynowego, nie udało się go uruchomić

#### Google Colab

Google Colab (z GPU T4)​ ograniczony jest do 12GB RAM systemowego, jest to problematyczne dla większych zdjęć, wymaga to odpowiedniego zarządzania pamięcią​. Na T4 występowaly częste losowe timeout'y, nie udało się uruchomić uczenia na dłużej niż 2.5h.

## Harmonogram

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

# Linki

- [Prezentacja prezentowana na ostatnich zajęciach](https://uam-my.sharepoint.com/:p:/r/personal/krzboj_st_amu_edu_pl/_layouts/15/doc.aspx?sourcedoc=%7Bab618fdf-5e98-4cf7-aa49-ea13fa807f4e%7D&action=edit)
- [Dokumentacja robocza, która zawiera całą historie prac i podejmowanych decyzji](https://uam-my.sharepoint.com/personal/krzboj_st_amu_edu_pl/_layouts/15/doc.aspx?sourcedoc={dc695bbe-68d1-4947-8c29-1d008f252a3b}&action=edit)

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