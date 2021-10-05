
# Associated_Learning

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

* For AGNews and DBpedia, dataset will be automatically donwloaded during the training.
* For SST-2, please download the dataset from [GLUE Benchmark](https://gluebenchmark.com/tasks) and put the files into `./data/sst2/`.

## Execution

We use json file for the configuration. Before running the code, please check [`hyperparameters.json`](configs/) and select proper parameters.

Then just simply run:

```bash
python -m associated_learning.main
```

## Citation
