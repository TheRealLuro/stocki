# model_training

Training framework for the stocki 1D-CNN.

| file | what it holds |
| --- | --- |
| `config.py` | all tunables, **including the input feature count** |
| `model.py` | `StockCNN1D` — the base model |
| `dataloader.py` | `StockDataLoader` — batching done, data reading stubbed |
| `train.py` | training loop, checkpointing, resume |
| `export_onnx.py` | checkpoint → `.onnx` for the API |
| `main.py` | CLI: `train` / `export` |

```bash
pip install -r requirements.txt

python main.py train                     # resumes from checkpoints/latest.pt if present
python main.py train --fresh --epochs 20 # ignore checkpoints, start from random weights
python main.py export                    # checkpoints/latest.pt -> artifacts/model.onnx
python model.py                          # print the architecture + a forward-pass shape check
```

## Before this can train

Two things are still open, both marked in the code:

1. **Feature count** — `config.NUM_INPUT_FEATURES` (currently a placeholder `8`). It is
   read by the model, the dataloader and the exporter, so it only needs changing there.
2. **Data reading** — `StockDataLoader._index_examples()` and `.get_example()` in
   `dataloader.py`. Everything else (shuffling, batching, collation, train/val split,
   shape validation) is built on top of `get_example`, so those two methods are the
   whole job. `train.py` fails with an explicit message until they exist.

## Tensor layout

```
input  (batch, NUM_INPUT_FEATURES, SEQUENCE_LENGTH)   channels-first, as Conv1d wants
output (batch, NUM_OUTPUTS)
```

The network pools over time, so a trained model accepts sequence lengths other than the
one it trained on; `sequence_length` is exported as a dynamic ONNX axis.

## Checkpoints

`train.py` writes `checkpoints/epoch_XXX.pt` at the end of every epoch and mirrors it to
`checkpoints/latest.pt`, which is the resume point. Each file carries the weights, the
optimizer state, the epoch number, the architecture config and that epoch's metrics — so
`export_onnx.py` can rebuild the right architecture from the checkpoint alone.
