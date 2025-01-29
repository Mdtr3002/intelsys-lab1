import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('diabetes.csv')
print(dataset.shape)
dataset.describe(include='all').round(6)

# Export to Markdown file
# description = dataset.groupby('Age').size()
# print(description)
# print(dataset.tail(3))
# with open('describe.md', 'w') as f:
#     f.write(description.to_markdown())
# print(dataset.groupby('Species').size())
# dataset.describe().to_csv('describe.csv', index=False)

# plt.figure()
# sns.pairplot(dataset, hue = "Outcome", size=3, markers=["o", "s", "D"])
# plt.show()

# plt.figure()
# dataset.boxplot(by="Outcome", figsize=(30, 20))
# plt.show()
# plt.savefig('boxplot.png')


# plt.figure()
# dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(30, 20))
# plt.show()

feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']
X = dataset[feature_cols].values
y = dataset['Outcome'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)

#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
#save to my_figure.png
plt.savefig('accuracy.png')

# Tạo model (k = 1)
classifier = KNeighborsClassifier(n_neighbors=6)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test)
# print(y_pred)

cm = confusion_matrix(y_test, y_pred)
# print(cm)

#Sử dụng hàm accuracy_score
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

acccuracy_byhand = (94+26)/(94+26+13+21)*100
print('Accuracy of our model is equal ' + str(round(acccuracy_byhand, 2)) + ' %.')

accuracy_knn = knn.score(X_test, y_test)*100
print('Accuracy of our model is equal ' + str(round(accuracy_knn, 2)) + ' %.')