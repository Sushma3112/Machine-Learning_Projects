from  sklearn.svm  import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score


data = load_breast_cancer()

X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = SVC(kernel='rbf',C=1, gamma='scale')
#rbf = Radial Basis Function   ### most powerful used asked kernal

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

print("Accuracy",accuracy_score(Y_pred,Y_test))
print("Confusion matrix",confusion_matrix(Y_pred,Y_test))


