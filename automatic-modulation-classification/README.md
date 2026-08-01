# Automatic Modulation Classification (AMC)

Deep-learning classifier that recognizes **BPSK, QPSK, 16QAM and FSK** from raw I/Q samples. The repository contains the full pipeline — signal + channel simulation, STFT spectrogram generation, several CNN architectures (1D I/Q CNN, baseline 2D CNN, ResNet18, EfficientNet-B0), training with LR scheduling and augmentation, and a structured robustness benchmark under noise, interference and frequency offsets.

---

## Highlights

- **98.6 % test accuracy / 0.986 macro-F1** with the raw-I/Q 1D CNN (10 epochs, CPU).
- Accuracy-vs-SNR curve, confusion matrix and per-class report produced automatically.
- Robustness benchmark sweeping SNR, co-channel interference, carrier frequency offset and phase noise.
- Three input representations selectable from the CLI: raw I/Q, magnitude STFT, complex (magnitude + real + imag) STFT.
- Fully synthetic, reproducible dataset — no external download required.

---

## Signal and channel model

`src/modulation.py` synthesizes each frame from random symbols, pulse-shaped with a root-raised-cosine filter, then passes it through a configurable channel:

| Impairment | Parameter |
|---|---|
| AWGN | `snr_db` (default sweep −6 → +20 dB) |
| Carrier frequency offset | `freq_offset` (normalized) |
| Static phase offset | `phase_offset` |
| Random-walk phase noise | `phase_noise_std` |
| Co-channel CW interference | `interference_db` |

The same code path works on recorded I/Q — pass a `(n_frames, frame_length)` `complex64` array to `ModulationSpectrogramDataset`.

---

## Project structure

```
├── src/
│   ├── modulation.py              # RRC pulse shaping, modulators, channel impairments
│   ├── spectrogram.py             # STFT (magnitude / complex), resizing, augmentation
│   ├── dataset.py                 # Torch Dataset, stratified split, dataloaders
│   ├── models.py                  # IQCNN1D, BaselineCNN, ResNet18, EfficientNet-B0
│   ├── train.py                   # Training loop (AdamW, cosine/step/onecycle schedules)
│   ├── evaluate.py                # Report, confusion matrix, accuracy-vs-SNR curve
│   ├── robustness.py              # Structured stress benchmarking
│   ├── compare_architectures.py   # Train several archs and tabulate results
│   └── predict.py                 # Single-frame inference (file or synthetic demo)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/LakhannB/automatic-modulation-classification.git
cd automatic-modulation-classification

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
# default: 1D CNN over raw I/Q
python -m src.train --epochs 20 --n-per-class 500

# spectrogram models
python -m src.train --arch baseline_cnn   --input-mode complex --epochs 20
python -m src.train --arch resnet18       --input-mode complex --epochs 20 --lr 5e-4
python -m src.train --arch efficientnet_b0 --input-mode complex --epochs 20 --lr 5e-4
```

Useful flags: `--scheduler {cosine,step,onecycle,none}`, `--lr`, `--batch-size`, `--label-smoothing`, `--n-fft`, `--hop-length`, `--image-size`, `--n-per-class`, `--pretrained`, `--seed`.

Artifacts land in `artifacts/<arch>/`: `best_model.pt`, `history.json`, `test_metrics.json`, `test_confusion_matrix.png`, `test_accuracy_vs_snr.png`, `summary.json`.

### Compare architectures

```bash
python -m src.compare_architectures --archs baseline_cnn resnet18 efficientnet_b0 --epochs 15
```

Trains each architecture under identical settings and writes `artifacts/comparison.json`.

### Robustness benchmark

```bash
python -m src.robustness --conditions snr interference freq_offset phase_noise --n-per-class 60
```

### Single-frame inference

```bash
python -m src.predict --demo QPSK --snr-db 12          # synthetic frame
python -m src.predict --iq-file capture.npy            # recorded complex64 samples
```

---

## Results

Reference run: `--arch iq_cnn --input-mode iq --epochs 10 --n-per-class 350` (CPU, 6 s/epoch).

```
test accuracy 0.9856   macro-F1 0.9856

              precision    recall  f1-score
        BPSK       1.00      1.00      1.00
        QPSK       0.96      0.98      0.97
       16QAM       0.98      0.96      0.97
         FSK       1.00      1.00      1.00
```

### Input representation matters

| Input | Model | Test accuracy | Macro-F1 |
|---|---|---|---|
| Magnitude STFT | Baseline 2D CNN | 0.572 | 0.570 |
| Complex STFT (mag + Re + Im) | Baseline 2D CNN | 0.514 | 0.497 |
| Raw I/Q (2 × 1024) | 1D CNN | **0.986** | **0.986** |

Magnitude spectrograms separate FSK cleanly but collapse BPSK / QPSK / 16QAM, which share an almost identical power spectrum and differ only in constellation phase and amplitude. Feeding the raw I/Q stream to a 1D CNN keeps that information and resolves the confusion — the spectrogram models are kept in the repo as the documented baseline for this comparison.

### Robustness (1D CNN, 25 frames per class per point)

| SNR | −10 dB | −5 dB | 0 dB | ≥ 5 dB |
|---|---|---|---|---|
| Accuracy | 0.28 | 0.59 | 0.91 | 1.00 |

| Carrier frequency offset | 0 | 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|---|
| Accuracy | 1.00 | 1.00 | 1.00 | 0.75 | 0.59 |

| CW interference (relative) | −20 dB | −15 dB | −10 dB | −5 dB | 0 dB |
|---|---|---|---|---|---|
| Accuracy | 1.00 | 1.00 | 0.97 | 0.63 | 0.31 |

Classification stays perfect down to 5 dB SNR and degrades gracefully below 0 dB. Frequency offsets above ~1 % of the sample rate and interference stronger than −5 dB are the two dominant failure modes — both are addressable by training with wider offset augmentation or adding a coarse frequency-estimation front end.

---

## Training configuration

| Setting | Value |
|---|---|
| Frame length | 1024 complex samples |
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | Cosine annealing (step / one-cycle also available) |
| Loss | Cross-entropy, label smoothing 0.05 |
| Augmentation | I/Q: time shift, random phase rotation, jitter · Spectrogram: time/frequency shift, SpecAugment masking |
| Split | Stratified 70 / 15 / 15 |
| Checkpointing | Best validation macro-F1 |

---

## Possible extensions

- Higher-order modulations (64QAM, 256QAM, OFDM) and multipath fading channels.
- Two-branch model fusing raw I/Q with the spectrogram.
- Coarse frequency/timing synchronization front end to widen the usable offset range.
- ONNX export and quantization for SDR edge deployment.

---

## Author

**Bijinepalli Lakhan** — [GitHub](https://github.com/LakhannB) · [LinkedIn](https://www.linkedin.com/in/lakhanb)
