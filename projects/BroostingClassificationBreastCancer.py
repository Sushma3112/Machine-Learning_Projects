import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


#-----------------------------------------------------
# Step 1:Load the dataset
#-----------------------------------------------------

df= pd.read_csv("breastCancer.csv")
print("Shape of dataset:",df.shape)
print("First 5 row:",df.head())

#-----------------------------------------------------
# Step 2:Seprate features and Label
#-----------------------------------------------------

X = df.drop("target",axis=1)
Y = df["target"]

#-----------------------------------------------------
# Step 3:Split dataset for training and testing
#-----------------------------------------------------

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=42,random_state=42)

#-----------------------------------------------------
# Step 4:Create Boosting Model #AdaBoost
#-----------------------------------------------------

boost_model = AdaBoostClassifier(
    n_estimators=50, #create model
    learning_rate=1.0,#weight increment 1 wrong model
    random_state=42
)
#-----------------------------------------------------
# Step 6:Train Boost Model
#-----------------------------------------------------

boost_model.fit(X_train,Y_train)

#-----------------------------------------------------
# Step 7:Test Boost Model
#-----------------------------------------------------

Y_pred=boost_model.predict(X_test)

#-----------------------------------------------------
# Step 8:Evaluate Boost Model
#-----------------------------------------------------

print("Boostin Accuracy:",accuracy_score(Y_pred,Y_test))
print("Confusion matrix:\n",confusion_matrix(Y_pred,Y_test))