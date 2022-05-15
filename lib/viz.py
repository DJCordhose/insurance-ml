# XXX having a library as an ordinary python file comes with all the merits of traditional software development

# https://github.com/parrt/dtreeviz
from dtreeviz import clfviz
import matplotlib.pyplot as plt
import seaborn as sns


def plot_decision_boundaries(model, X, y_true, x1_range=None, x2_range=None):
    _, ax = plt.subplots(figsize=(8, 4), dpi=300)

    ranges = None
    if x1_range and x2_range:
        ranges = (x1_range, x2_range)

    clfviz(
        model, X, y_true,
        show=['instances', 'boundaries', 'probabilities', 'misclassified'],
        markers=['v', '^', 'd'],
        ntiles=50,
        ax=ax,
        ranges=ranges,
        tile_fraction=1.0,
        boundary_markersize=1.0,
        feature_names=["Age", "Max Speed"],
        colors={'class_boundary': 'black',
                'tile_alpha': 0.5,
                #  'warning' : 'yellow',
                'classes':
                [None,  # 0 classes
                 None,  # 1 class
                 None,  # 2 classes
                 ['#FF8080', '#FFFF80', '#8080FF'],  # 3 classes
                 ]
                }
    )


def plot_correlation_matrix(cm, labels=['speed', 'age', 'miles'], title='Correlation Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.title(title)

    hm = sns.heatmap(cm,
                     cbar=True,
                     annot=True,
                     square=True,
                     cmap=cmap,
                     fmt='.2f',
                     yticklabels=labels,
                     xticklabels=labels)
    return hm


def plot_metric_over_time(title, metric, validation, legend=['Training', 'Validation'], x_label='epochs', y_label='metric', log=True):
  if log:
    plt.yscale('log')
  plt.ylabel(y_label)
  plt.xlabel(x_label)
  plt.title(title)

  plt.plot(metric)
  plt.plot(validation)

  plt.legend(legend)
