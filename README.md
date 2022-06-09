# Notion page

- [meta-RL notion page](https://www.notion.so/nrfteams/meta-RL-b8e1176addfa46428a233d4f3d48e5d1) (private)


# Requirements

```
conda: (channel: pytorch, conda-forge)
	- numpy
	- scipy
	- matplotlib
	- ipython
	- ipdb
	- pytorch
	- torchvision
	- torchaudio
	- cvxpy
	- ffmpeg
	- pip:
		- loguru
```

# How to Run the Scripts

First, you should obtain transition data using
```bash
python preprossess.py
```

To train, run:
```bash
python train.py
```

To test, run:
```bash
python test.py
```

## The performance of the encoder

This section demonstrates the performance of the encoder.

Run
```bash
python test-encoder.py
```

## The effect of communicating the meta-learning layer and the RL agent


## Results of a single RL on multiple tasks

Runf
```bash
python examples/single-RL/main.py
```


## Comparison End-to-End Results

This seciton demonstrates the performance of the proposed FTC scheme.
The following methods are compared.
* LQR-EHS: use an exact hover solution (EHS), no FDI.
* LQR-NHS: use a near-hover solution (NHS), no FDI.
* SMC-EHS: use an EHS, no FDI.
