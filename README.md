# Thesis Project

This project provides a first runnable baseline for the thesis workflow:

- Centralized training on `data/global.csv`
- Federated training with 5 clients from `data/federated_clients/client_*.csv`
- Centralized TabTransformer-style training for the proposal's transformer baseline
- Shared evaluation with accuracy, precision, recall, F1, and macro AUC

The core baseline uses only `numpy` and `pandas` so it can run in the current environment without installing heavy deep learning frameworks first. The TabTransformer path requires `torch`.

## Project structure

```text
src/
  ciciot/
    config.py
    data.py
    metrics.py
    models/
      mlp_numpy.py
      tabtransformer_torch.py
train_centralized.py
train_federated.py
train_tabtransformer.py
```

## Dataset assumptions

- The label column is named `Label`
- Features are already numeric
- Your federated split exists under `data/federated_clients/`

## Run centralized training

```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_centralized.py
```

## Run federated training

```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_federated.py --test-rows 100000 --client-max-rows 50000 --rounds 10 --local-epochs 2
```

## Run centralized TabTransformer

Install PyTorch first, then run:

```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_tabtransformer.py --max-rows 200000 --epochs 20 --batch-size 512
```

Because the prepared CICIoT2023 features are numeric, this implementation uses learned per-feature token embeddings plus a transformer encoder. That is a practical TabTransformer-style baseline for this thesis codebase.

## Notes

- Start with row limits because CICIoT2023 is large and this machine has 16 GB RAM.
- Randomized sampling is more correct than taking the first `n` rows, but it is slower because the code scans the CSV in chunks.
- The current code is the lightweight baseline stage of the thesis plan.
- The centralized MLP path is now locked to the thesis multiclass setup: 8 encoded classes from the prepared dataset.
- The federated MLP path now defaults to the same `512 256 128` architecture as the best centralized MLP baseline for fair comparison.
- The next steps after the current MLP baselines are federated TabTransformer, differential privacy, scalability sweeps, and zero-day evaluation.
