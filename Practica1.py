import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split

trayectories = pd.read_csv("./ur5_trajectory_dataset.csv").dropna()
X = pd.get_dummies(trayectories.drop(columns="trajectory_class"),dtype=int)
y = trayectories["trajectory_class"]


# A partir de aqui es el arbol
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0, stratify=y)

tree = DecisionTreeClassifier(max_leaf_nodes=32,criterion="gini",random_state=1,min_samples_leaf=20,class_weight="balanced")
tree.fit(X_train,y_train)

plt.figure(figsize=(12, 6))
plot_tree(tree,class_names=tree.classes_.astype(str),impurity=True,precision=2)
plt.show()

# Esto lo he sacado de otro ejercicio, mirar como buscarlo en el scikit learn pq no lo tendre en el examen
importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values()
importance = importance[importance > 0]
print(importance.sort_values(ascending=False))