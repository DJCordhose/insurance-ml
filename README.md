# Solve insurance risk groups using machine learning

## Running the tutorial locally

### Dependencies

The workshop requires the following packages:

- scikit-learn
- pandas
- seaborn
- jupyter
- tensorflow
- numpy
- matplotlib
- evidently
- prometheus-client

### Install via Docker

For this workshop you will need [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) running on your machine. *(on mac os docker-compose is by default installed with Docker)*


### Local install

Installation via Docker is sufficient, buy local installation of a Python environment might be convenient. Local installation is sufficient if you don't want to run [Prometheus](https://prometheus.io) or [Grafana](http://grafana.com). 

We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

You can create an `mlops-workshop` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate mlops-workshop
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```
