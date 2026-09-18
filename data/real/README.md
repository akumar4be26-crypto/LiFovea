# Real-world lidar data

Place labeled `.npz` scans in these directories:

```text
data/real/
  train/
    scan_0001.npz
  val/
    scan_1001.npz
```

Each archive must contain:

- `xyz`: float array with shape `(N, 3)` in the sensor frame
- `label`: integer array with shape `(N,)`, using the project classes:
  - `0` drivable
  - `1` rough terrain
  - `2` static obstacle
  - `3` dynamic object
- `intensity`: optional float array with shape `(N,)`

Train with the simulator and real scans mixed together:

```bash
PYTHONPATH=. /usr/local/bin/python3 -m avr25d.models.train \
  --real-data data/real/train \
  --real-val-data data/real/val \
  --epochs 14 \
  --out checkpoints/avr25d_real.pt
```

Do not commit downloaded datasets unless their license permits redistribution.
Keep validation locations or drives separate from training locations.

For a SemanticKITTI checkout, convert standard `.bin` and `.label` files with:

```bash
PYTHONPATH=. /usr/local/bin/python3 tools/convert_semantickitti.py \
  /path/to/SemanticKITTI --sequences 00 01 02 --out data/real/train --every 2
```

The converter preserves `xyz` and intensity, maps SemanticKITTI labels into
the four project classes, and writes compressed `.npz` scans.

For nuScenes mini, weak labels can be generated from annotated 3D boxes:

```bash
PYTHONPATH=. /usr/local/bin/python3 tools/convert_nuscenes_boxes.py \
  data/real/nuscenes/extracted --out data/real/nuscenes/weak_npz
```

Only points inside annotated dynamic/static boxes are labeled. Other points use
`255` (unknown) and are ignored by the training loss and validation metrics.
