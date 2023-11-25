import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

label_Encoder = LabelEncoder()

dados = arff.loadarff('example.arff') #Rename this to match the name of your .arff file
data_Frame = pd.DataFrame(dados[0])
data_Frame = data_Frame.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

variavel_X = data_Frame[data_Frame.columns[:-1]]
variavel_Y = data_Frame[data_Frame.columns[-1]]

x_Codificado = variavel_X.apply(label_Encoder.fit_transform)
y_Codificado = label_Encoder.fit_transform(variavel_Y)
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x_Codificado, y_Codificado)

interface_grafica, axis = plt.subplots(figsize=(12, 7))
tree.plot_tree(clf, 
                feature_names=data_Frame.columns[:-1],
                filled=True,
                ax=axis)
plt.show()