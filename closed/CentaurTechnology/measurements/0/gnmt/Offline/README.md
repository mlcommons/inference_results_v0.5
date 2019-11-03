# 1. Problem
- This problem uses recurrent neural network to do language translation.
- The steps to train the model and generate the dataset are listed in [train_gnmt.txt](https://github.com/mlperf/inference/blob/master/v0.5/translation/gnmt/tensorflow/train_gnmt.txt). Basically, they follow the MLPerf training code. However, you can download the model and dataset with the scripts in this directory.

# 2. Directions

### Download Dataset / trained model

Download the dataset and the trained model per MLPerf provided scripts:

```bash
$ ./download_dataset.sh
$ ./download_trained_model.sh
$ ./verify_dataset.sh
```

### Install Dependencies

- Create python3.6 virtual environment

```
$ python3.6 -m venv env
$ source env/bin/activate
```

- Install dependencies with `pip`

```bash
$ python -m pip install -r requirements/requirements-py3.txt
```

- Evaluate accuracy to ensure the target BLEU.

3.  Run:

- Run the Offline Scenario in Performance mode:

```
python loadgen_gnmt.py --mode Performance
```

- Run the Offline Scenario in Accuracy Mode (Expected BLUE Score: 23.9)

```
python loadgen_gnmt.py --mode Accuracy
python process_accuracy.py
```
