# How Stage 1 Feeds Stage 2

Stage 1 writes per-frame landmark CSVs into `Feature_Data/`.

Required CSV columns:

```text
video, frame, timestamp_ms, num_hands
hand0_label, hand0_score, hand0_0 ... hand0_62
hand1_label, hand1_score, hand1_0 ... hand1_62
```

Stage 2 `build_windows.py` reads these CSVs and converts them into `.npz` windows. It uses `hand*_label` to create the canonical motion feature order:

```text
right hand 63 + left hand 63 = 126
```

Recommended command after Stage 1 finishes:

```bash
cd "../av_GRU_autoencoder"

python build_windows.py \
  --input-dir "../Feature_Data" \
  --output-dir "../Window_Data" \
  --sequence-length 60 \
  --hop-length 30 \
  --audio-dim 1 \
  --audio-quality 0.0
```

If the CSVs were made by an older extractor with no `hand*_label` columns, Stage 2 falls back to `hand0=right` and `hand1=left`. Re-extract with the updated Stage 1 scripts when two-hand identity matters.
