[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Protected  classification
This library contains the Python implementation of Protected probabilistic classification. The method is way of protecting probabilistic prediction models against changes in the data distribution, concentrating on the case of classification. This is important in applications of machine learning, where the quality of a trained prediction algorithm may drop significantly in the process of its exploitation under the presence of various forms of dataset shift.  

### Installation
```commandline
pip install protected-classification
```
The algorithm can be applied on top of an underlying scikit-learn algorithm for binary and multiclass classification problems.
### Usage
```commandline
from protected_classification import ProtectedClassification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
import numpy as np

np.random.seed(1)

X, y = make_classification(n_samples=1000, n_classes=2, n_informative=10, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
p_test = clf.predict_proba(X_test)

# Initialise Protected classification
pc = ProtectedClassification(estimator=clf)

# Calibrate test output probabilities
pc.fit(X_train, y_train)
p_prime = pc.predict_proba(X_test)

# Compare log loss of underlying RF algorithm and Protected classification
print('Underlying classifier log_loss (no dataset shift) ' + f'{log_loss(y_test, p_test):.3f}')
print('Protected classification log loss (no dataset shift) ' + f'{log_loss(y_test, p_prime):.3f}')

#  Assume a dataset shift where a random portion of the class labels is set to a single class
y_test[:100] = 0
ind = np.random.permutation(len(y_test))
X_test = X_test[ind]
y_test = y_test[ind]    

p_test = clf.predict_proba(X_test)

# Generate protected output probabilities  (assuming that test examples arrive sequentially)
pc = ProtectedClassification(estimator=clf)
p_prime = pc.predict_proba(X_test, y_test)

# Compare log loss of underlying RF algorithm and Protected classification
print('Underlying classifier log_loss (dataset shift) ' + f'{log_loss(y_test, p_test):.3f}')
print('Protected classification log loss (dataset shift) ' + f'{log_loss(y_test, p_prime):.3f}')
```

### Examples
Further examples can be found in the github repository https://github.com/ip200/protected-calibration in the *examples* folder:

- [simple_classification.ipynb](https://github.com/ip200/protected-classification/blob/main/notebooks/simple_classification.ipynb) for an example of the method appplied to calibrate the outputs of the underlying algorithm
- [protected_classification.ipynb](https://github.com/ip200/protected-classification/blob/main/notebooks/protected_classification.ipynb) for an example of the method used to protect the underlying algorithm under various forms of dataset shift

### Citation
If you find this library useful please consider citing:

- Vovk, Vladimir, Ivan Petej, and Alex Gammerman. "Protected probabilistic classification." In Conformal and Probabilistic Prediction and Applications, pp. 297-299. PMLR, 2021. (arxiv version https://arxiv.org/pdf/2107.01726.pdf)
