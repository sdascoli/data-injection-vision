# Readme

To reproduce the figures where the reservoir dataset is used, run

```python main.py --dataset_type $DATASET_TYPE --alpha $ALPHA```

with the desired values of the transfer parameter ```$ALPHA```.

To reproduce the figures where the reservoir dataset is not used, run

```python main.py --dataset_type $DATASET_TYPE --num_reservoir 0```

For the various other training parameters, see the argument parser in ```main.py```.
