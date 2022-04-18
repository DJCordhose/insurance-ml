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
- mlflow

### Install via Docker

For this workshop you will need [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) running on your machine. *(on mac os and Windows docker-compose is by default installed with Docker)*

#### Run the application, Prometheus and Grafana in Docker

To build the application Docker image and start the application container as well as Prometheus and Grafana together, run the following command (from the root of this repo):

``` sh
docker-compose up --build
```

*If you see errors it may be because you still have the previous version of the application running and therefore might be using the same port as you are now trying to access with Docker. Or the ports interfere with local installations. A local Grafana installation probably runs on port 3000 as a service.*

You should then be able to access the Prometheus dashboard on `http://localhost:9090` and Grafana on `http://localhost:3000`.

### Local install

Installation via Docker is sufficient, but local installation of a [Python](https://www.python.org/downloads/) environment might be convenient. Local installation is sufficient if you don't want to run [Prometheus](https://prometheus.io) or [Grafana](http://grafana.com). 

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
