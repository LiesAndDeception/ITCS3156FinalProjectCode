import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Peter Crean - 801365057 - ITCS 3156 Intro to Machine Learning Spring 2025
# Final Project Report Code




# DATA LOADING

# Load .csv into Dataframe
diabetes_dataset_df = pd.read_csv("diabetes_dataset.csv")

# Drop unnamed index column
diabetes_dataset_df.drop('Unnamed: 0', axis=1, inplace=True)
diabetes_dataset_df.info()


# According to the ADA (American Diabetes Association), diabetes is diagnosed when Fasting Blood Glucose is above 126 mg/dL or HpA1C is above 6.5%.

# Create target column based on FastingBloodGlucose or HbA1c column information:
diabetes_dataset_df['Target'] = (diabetes_dataset_df['HbA1c'] >= 6.5) | (diabetes_dataset_df['Fasting_Blood_Glucose'] >= 126)
# Convert bool vals to binary integers:
diabetes_dataset_df["Target"] = diabetes_dataset_df['Target'].astype(int)
print(f"\nDiabetes dataset with target column: \n{diabetes_dataset_df}")

# uncomment below for .info() view:
#diabetes_dataset_df.info()




# VISUALIZATION (of numerical values)
import seaborn as sns

diabetes_col_data = ["Age", "Fasting_Blood_Glucose", "HbA1c"]
sns.pairplot(diabetes_dataset_df[diabetes_col_data + ['Target']], hue='Target')
plt.show()




# DATA PREPROCESSING

# One-hot encode categorical data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

categorical_columns = diabetes_dataset_df.select_dtypes(include=['object']).columns.tolist()
ohe = OneHotEncoder(sparse_output=False)
encoded_df = ohe.fit_transform(diabetes_dataset_df[categorical_columns])
intermediate_encoded = pd.DataFrame(encoded_df, columns=ohe.get_feature_names_out(categorical_columns))
diabetes_ohe = pd.concat([diabetes_dataset_df, intermediate_encoded], axis=1)
diabetes_ohe = diabetes_ohe.drop(categorical_columns, axis=1)

print(f"\nOne-hot encoded diabetes dataframe: \n{diabetes_ohe}")

# uncomment below for .info() view:
#diabetes_ohe.info()

# Split data into training, validation, and testing sets

# Convert dataframes to NumPy arrays
X = diabetes_ohe.drop('Target', axis=1).values
y = diabetes_ohe['Target'].values

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, train_size=.8, random_state=42)
X_trn, X_vld, y_trn, y_vld = train_test_split(X_trn, y_trn, train_size=.8, random_state=42)

# Standardize training set and fit to other sets
scale = StandardScaler()
scale.fit(X_trn)
X_trn, X_vld, X_tst = scale.transform(X_trn), scale.transform(X_vld), scale.transform(X_tst)




# LOGISTIC REGRESSION (with L2 penalty)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_trn, y_trn)




# RANDOMFORESTCLASSIFIER
from sklearn.ensemble import RandomForestClassifier
randfrst = RandomForestClassifier()
randfrst.fit(X_trn, y_trn)




# NEURAL NETWORK
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(8,4), max_iter=1000, random_state=42)
clf.fit(X_trn, y_trn)




# METRICS

# Get validation predictions:
lr_y_pred = logreg.predict(X_vld)
rf_y_pred = randfrst.predict(X_vld)
clf_y_pred = clf.predict(X_vld)

# Calculate accuracy metrics:
print(f"\nLogisticRegression Validation Accuracy: {accuracy_score(y_vld, lr_y_pred) * 100}%")
print(f"RandomForest Validation Accuracy: {accuracy_score(y_vld, rf_y_pred) * 100}%")
print(f"NeuralNetwork Validation Accuracy: {accuracy_score(y_vld, clf_y_pred) * 100}%")

# Get test predictions:

lr_y_tst_pred = logreg.predict(X_tst)
rf_y_tst_pred = randfrst.predict(X_tst)
clf_y_tst_pred = clf.predict(X_tst)

# Print test metrics/reports:

print(f"\nLogisticRegression Test Classification Report: \n{classification_report(y_tst, lr_y_tst_pred)}")
print(f"LogisticRegression Test Accuracy: {accuracy_score(y_tst, lr_y_tst_pred) * 100}%")

print(f"\nRandomForest Test Classification Report: \n{classification_report(y_tst, rf_y_tst_pred)}")
print(f"RandomForest Test Accuracy: {accuracy_score(y_tst, rf_y_tst_pred) * 100}%")

print(f"\nNeuralNetwork Test Classification Report: \n{classification_report(y_tst, clf_y_tst_pred)}")
print(f"NeuralNetwork Test Accuracy: {accuracy_score(y_tst, clf_y_tst_pred) * 100}%")

# Print test confusion matrices:

# LogisticRegression cm:
lr_cm = confusion_matrix(y_tst, lr_y_tst_pred)
cmdisp = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=logreg.classes_)
cmdisp.plot()
plt.title('LogisticRegression Confusion Matrix')
plt.show()

# RandomForest cm:
rf_cm = confusion_matrix(y_tst, rf_y_tst_pred)
cmdisp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=randfrst.classes_)
cmdisp.plot()
plt.title('RandomForest Confusion Matrix')
plt.show()

# NeuralNetwork cm:
clf_cm = confusion_matrix(y_tst, clf_y_tst_pred)
cmdisp = ConfusionMatrixDisplay(confusion_matrix=clf_cm, display_labels=clf.classes_)
cmdisp.plot()
plt.title('NeuralNetwork Confusion Matrix')
plt.show()