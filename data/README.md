# data/

| Folder | Contents |
|--------|----------|
| `videos/` | Raw input recordings (`press_train.mp4`, `press_test.mp4`) |
| `tracks/` | Per-frame measurements extracted by `src/track.py`, plus overlay videos and contact sheets for checking the tracking |

Everything in `tracks/` can be regenerated from `videos/`:

```bash
python src/track.py train
python src/track.py test
```
