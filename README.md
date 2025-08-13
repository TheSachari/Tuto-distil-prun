# tuto_model_shrink (modulaire)

Ce projet démontre comment appliquer **distillation**, **pruning**, **quantization** et **compression** sur un CNN PyTorch (MNIST), avec un code **découpé en modules** et des **toggles CLI** pour activer/désactiver chaque étape.

## Installation rapide
```bash
pip install torch torchvision torchaudio  # ou version CUDA adaptée à votre GPU
```

## Lancer (toutes les étapes par défaut)
```bash
python -m tuto_model_shrink.main   --epochs 2   --qat_epochs 2   --prune_ratio 0.5   --batch 128   --data_dir ./data   --out_dir ./artifacts
```

## Choisir précisément les méthodes (flags ON/OFF)
Chaque étape peut être activée/désactivée via des booléens `--flag` / `--no-flag` :
- `--teacher / --no-teacher` : entraîner le teacher (par défaut: ON)
- `--student_base / --no-student_base` : entraîner un student de base (référence) (par défaut: OFF)
- `--distill / --no-distill` : distillation teacher→student (par défaut: ON)
- `--prune / --no-prune` : pruning du student (par défaut: ON)
- `--dynquant / --no-dynquant` : quantification dynamique INT8 (CPU) (par défaut: ON)
- `--qat / --no-qat` : QAT + conversion INT8 (CPU) (par défaut: ON)
- `--fp16 / --no-fp16` : export des poids FP16 (par défaut: ON)
- `--gzip / --no-gzip` : sauvegarder aussi un `.pth.gz` (par défaut: ON)

### Exemples
- Distillation + DynQuant seulement :
```bash
python -m tuto_model_shrink.main --no-prune --no-qat
```
- Pruning uniquement (sans distillation) :
```bash
python -m tuto_model_shrink.main --no-distill --prune --prune_ratio 0.7
```
- Désactiver toute quantization :
```bash
python -m tuto_model_shrink.main --no-dynquant --no-qat --no-fp16
```

## Sorties
Les variantes et un `report.json` sont écrits dans `--out_dir` (par défaut `./artifacts`).
