# id:7-7--7-0 
# id:7-14--7-0 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc

df = pd.read_csv('ML-4-DS1.csv', skiprows=1, names=['x1', 'x2', 'y'])
print(df.head())
print(df.shape)

X = df[['x1', 'x2']].values
y = df['y'].values

X1 = df[df['y'] == 1]
X2 = df[df['y'] == -1]

# scatter plot for data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X1['x1'], X1['x2'], marker='+', color='g', label='+1')
ax.scatter(X2['x1'], X2['x2'], marker='o', color='b', s=10, label='-1')
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.legend(loc='best')
plt.title("Scatter plot of data")
"""plt.show()"""
plt.savefig(f'Scatter_plot.png')
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
C_val = [1, 2, 3, 4, 5, 10, 15]
poly_range = [1, 2, 3, 4]

best_score = 0.0
best_c = None
best_degree = None
best_model = None

result=[]
for degree in poly_range:
    accuracy = []
    accuracy_std = []
    for C in C_val:
        poly_features = PolynomialFeatures(degree)
        x_train_poly = poly_features.fit_transform(X_train)
        
        scaler = StandardScaler()
        x_train_poly = scaler.fit_transform(x_train_poly)
        
        model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000)
        #cross validation
        scores = cross_val_score(model, x_train_poly, y_train, cv=5, scoring='accuracy')

        accuracy.append(scores.mean())
        accuracy_std.append(scores.std())
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_c = C
            best_degree = degree
            best_model = model
            best_poly_features = poly_features
            best_scaler = scaler

    result.append((degree, accuracy, accuracy_std))
print('Best C:', best_c)
print('Best Degree:', best_degree)
print('Best Score:', best_score)
for degree, accuracy, accuracy_std in result:
    plt.errorbar(C_val, accuracy, yerr=accuracy_std, label=f'Degree = {degree}')

plt.xlabel('C')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Relationships among Polynomial Degree, C and Accuracy')
"""plt.show()"""
plt.savefig(f'Relationship-betn-deg-c.png')
plt.close()

X_train_best = best_poly_features.transform(X_train)
X_train_best = best_scaler.transform(X_train_best)
best_model.fit(X_train_best, y_train)  

X_test_poly = best_poly_features.transform(X_test)
X_test_poly = best_scaler.transform(X_test_poly)
y_pred = best_model.predict(X_test_poly)

conf_mat_lr = confusion_matrix(y_test, y_pred)
#KNN
k_val = range(1, 100)
accuracy_scores = []
accuracy_std = []
for k in k_val:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())
    accuracy_std.append(scores.std())

best_k = k_val[np.argmax(accuracy_scores)]
best_score = max(accuracy_scores)
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
y_pred = final_knn.predict(X_test)

print('Best k:', best_k)
print('Best Cross-Validation Score:', best_score)
plt.errorbar(k_val, accuracy_scores, yerr=accuracy_std)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('kNN Classifier: Accuracy vs. k')
"""plt.show()"""
plt.savefig(f'KNN-accurracy.png')
plt.close()



#plot for confusion matrix for KNN and LR

conf_mat_knn = confusion_matrix(y_test, y_pred)
# Plot for confusion matrix
conf_df = pd.DataFrame(conf_mat_lr, columns=['Predcit Yes','Predcit No'], index = ['Actual Yes','Actual No'])
labels = np.array([["TN", "FP"], ["FN", "TP"]])
labels = (np.asarray(["{}\n\n{}".format(label, value) for label, value in zip(labels.flatten(), conf_mat_lr.flatten())])
              ).reshape(2, 2)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_df, annot=labels, fmt="", cmap='Blues', cbar=False, annot_kws={"size": 16})
plt.title('Confusion Matrix for LR')
"""plt.show()"""
plt.savefig(f'LR_ConfusionMatrix.png')
plt.close()
conf_df = pd.DataFrame(conf_mat_knn, columns=['Predcit Yes','Predcit No'], index = ['Actual Yes','Actual No'])
labels = np.array([["TN", "FP"], ["FN", "TP"]])
sns.heatmap(conf_df, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion Matrix for KNN')
"""plt.show()"""
plt.savefig(f'KNN_ConfusionMatrix.png')
plt.close()

dummy_most_frequent = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_pred_most_frequent = dummy_most_frequent.predict(X_test)
conf_mat_most_frequent = confusion_matrix(y_test, y_pred_most_frequent)
dummy_random = DummyClassifier(strategy='uniform', random_state=42).fit(X_train, y_train)
y_pred_random = dummy_random.predict(X_test)
conf_mat_random = confusion_matrix(y_test, y_pred_random)

plt.figure(figsize=(6, 5))
plt.plot(1, 3, 1)
sns.heatmap(conf_mat_most_frequent, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion matrix for Baseline (Most Frequent) model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
"""plt.show()"""
plt.savefig(f'MF_ConfusionMatrix.png')
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(1, 3, 2)
sns.heatmap(conf_mat_random, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion matrix for Baseline (Random) model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
"""plt.show()"""
plt.savefig(f'RM_ConfusionMatrix.png')
plt.close()

# Logistic Regression
y_scores_logistic = best_model.predict_proba(X_test_poly)[:, 1]
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_scores_logistic)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

# kNN
y_scores_knn = final_knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_scores_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Baseline: Most Frequent
y_scores_most_frequent = np.full_like(y_test, dummy_most_frequent.predict(X_test)[0])
fpr_most_frequent, tpr_most_frequent, _ = roc_curve(y_test, y_scores_most_frequent)
roc_auc_most_frequent = auc(fpr_most_frequent, tpr_most_frequent)

# Baseline: Random
y_scores_random = dummy_random.predict_proba(X_test)[:, 1]
fpr_random, tpr_random, _ = roc_curve(y_test, y_scores_random)
roc_auc_random = auc(fpr_random, tpr_random)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_logistic, tpr_logistic, label='Logistic Regression (area = {:.2f})'.format(roc_auc_logistic))
plt.plot(fpr_knn, tpr_knn, label='kNN (area = {:.2f})'.format(roc_auc_knn))
plt.plot(fpr_most_frequent, tpr_most_frequent, label='Most Frequent (area = {:.2f})'.format(roc_auc_most_frequent))
plt.plot(fpr_random, tpr_random, label='Random (area = {:.2f})'.format(roc_auc_random))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for all models')
plt.legend(loc="lower right")
"""plt.show()"""
plt.savefig(f'ROC plot.png')
plt.close()






