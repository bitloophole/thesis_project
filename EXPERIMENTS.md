# Experiment Design

This file defines the recommended order for experiments so results stay comparable and thesis-ready.

## 1. Centralized MLP baseline

Use a randomized sample, stratified split, class weighting, and standardization.

```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_centralized.py
```

Record:

- Accuracy
- Precision macro
- Recall macro
- F1 macro
- AUC macro OVR
- Training time

Suggested stronger centralized variants:

- Weighted cross-entropy:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_centralized.py --max-rows 50000 --epochs 20 --batch-size 256
```

- Focal-loss style variant for rare classes:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_centralized.py --max-rows 50000 --epochs 20 --batch-size 256 --focal-gamma 1.5
```

- Higher-row run with balanced oversampling of the training split:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_centralized.py --max-rows 150000 --epochs 20 --batch-size 256 --hidden-dims 512 256 128 --oversample-train --oversample-target-fraction 0.5
```

- Larger high-row run if memory and runtime stay stable:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_centralized.py --max-rows 200000 --epochs 20 --batch-size 256 --hidden-dims 512 256 128 --oversample-train --oversample-target-fraction 0.5
```

- Small sweep over centralized MLP settings:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe sweep_centralized.py --max-trials 6 --max-rows 20000 --epochs 10 --batch-size 256
```

## 2. Federated 5-client MLP baseline

Use the same model family and preprocessing discipline so the comparison is fair.

```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_federated.py --test-rows 100000 --client-max-rows 50000 --rounds 10 --local-epochs 2 --batch-size 256
```

Record:

- Final global accuracy
- Final global precision macro
- Final global recall macro
- Final global F1 macro
- Final global AUC macro OVR
- Training time
- Per-round metric trend

## 3. Centralized TabTransformer baseline

This path requires `torch`.

```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_tabtransformer.py --max-rows 200000 --epochs 20 --batch-size 512
```

Recommended first TabTransformer sweep:

- Wider token representation:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_tabtransformer.py --max-rows 200000 --epochs 20 --batch-size 512 --d-token 96 --n-heads 8 --n-layers 4 --ffn-dim 192 --mlp-hidden-dim 192
```

- Deeper encoder:
```powershell
C:\Users\kk199\AppData\Local\Programs\Python\Python313\python.exe train_tabtransformer.py --max-rows 200000 --epochs 20 --batch-size 512 --d-token 64 --n-heads 8 --n-layers 6 --ffn-dim 128
```

Record the same metrics as the MLP baseline and compare directly against the best centralized MLP configuration.

## 4. Controlled ablations

Run each change one at a time:

- Without class weighting
- Without standardization
- With focal gamma `1.5`
- With train oversampling at `0.5`
- With train oversampling at `1.0`
- Different weight decay
- Non-stratified split
- Different hidden dimensions
- Different local epochs
- Different communication rounds

## 5. Known data concern

The current federated client files appear highly skewed. Sampled label distributions showed several clients dominated by one label, which means the federated setup is strongly non-IID. That is useful for later thesis analysis, but it can depress the first federated results.

## 6. Next implementation priorities

1. Add federated TabTransformer so the transformer baseline can be compared fairly with federated MLP.
2. Add differential privacy in the federated update path.
3. Add client-count scalability experiments.
4. Add zero-day attack evaluation.
