from TFFusions.SecondLevel.GCForest import gcForest
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading the data
iris = load_iris()
X = iris.data
y = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)

gcf = gcForest(shape_1X=4, window=2, tolerance=0.2)
gcf.fit(X_tr, y_tr)

pred_X = gcf.predict(X_te)
print(pred_X)

gcf.predict_proba(X_te)

# loading the data
digits = load_digits()
X = digits.data
y = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4)

gcf = gcForest(shape_1X=[8,8], window=[4,6], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7)
gcf.fit(X_tr, y_tr)
