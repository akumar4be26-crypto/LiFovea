# AVR-2.5D benchmark

- backend: `torch:pointnet2lite`
- frames: 20
- point accuracy: **0.954**, mIoU **0.827**
- network-only accuracy: 0.964 (mIoU 0.840)
- end-to-end latency: 229.8 ms (4.4 fps)
- map: 9.88 MB, 97x smaller than a uniform 5 cm grid

## Per class

| class | IoU | recall | precision |
|---|---:|---:|---:|
| drivable | 0.905 | 0.909 | 0.996 |
| rough_terrain | 0.885 | 0.992 | 0.891 |
| static_obstacle | 0.970 | 0.992 | 0.977 |
| dynamic_object | 0.548 | 0.582 | 0.904 |

## By range

| band | points | accuracy | mIoU |
|---|---:|---:|---:|
| 0-10 m | 822,872 | 0.953 | 0.840 |
| 10-20 m | 186,864 | 0.973 | 0.700 |
| 20-40 m | 38,342 | 0.884 | 0.709 |
| 40-70 m | 9,379 | 0.908 | 0.796 |
| 70-100 m | 3,270 | 0.926 | 0.695 |

## Elevation error vs ground truth

| band | median cell | RMSE | P95 abs | cells |
|---|---:|---:|---:|---:|
| 0-10 m | 5 cm | 18 mm | 10 mm | 94,106 |
| 10-20 m | 10 cm | 42 mm | 22 mm | 25,712 |
| 20-40 m | 20 cm | 35 mm | 38 mm | 12,933 |
| 40-70 m | 40 cm | 261 mm | 77 mm | 5,096 |
| 70-100 m | 80 cm | 141 mm | 147 mm | 1,350 |

## Latency

| stage | mean ms | p95 ms |
|---|---:|---:|
| features | 16.1 | 17.5 |
| inference | 118.4 | 130.1 |
| projection | 56.5 | 66.4 |
| traversability | 38.8 | 48.8 |
| total | 229.8 | 259.1 |

## Structural integrity

| invariant | value |
|---|---:|
| points_mapped | 1 |
| leaf_level_consistency | 1 |
| cross_level_overlaps | 0 |
| ring_violations | 0 |
| alignment_error_m | 0 |
