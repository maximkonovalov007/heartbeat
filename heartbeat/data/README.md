# Recordings

Raw CSV straight off the board: `t_us` (device microseconds), then x/y/z in g for
each of the three accelerometers (`f1` = level 1, `f2` = level 2, `f3` = roof),
then a cumulative I2C error count. 200 samples per second.

| File | Tower state | What it is |
|---|---|---|
| `baseline.csv` | all tight | the reference capture |
| `floor1.csv` | level 1 loose | training capture |
| `floor2.csv` | level 2 loose | training capture |
| `roof.csv` | roof loose | training capture |
| `retight1.csv`, `retight2.csv` | all tight, retightened by hand | repeatability check |
| `tight2.csv` | all tight, after finding a joint had worked loose | the re-tightened check |
| `transition.csv` | tight, then level 1 loose at 45 s | spliced, for testing how fast it switches |
| `recovery.csv` | level 1 loose, then tight at 45 s | the same in reverse |
| `quiet.csv` | all tight | nobody touching it — ambient only |
| `quiet_f1.csv` | level 1 loose | nobody touching it — ambient only |

Any of them can drive the console without the hardware:

```bash
python monitor.py --replay data/roof.csv
```
