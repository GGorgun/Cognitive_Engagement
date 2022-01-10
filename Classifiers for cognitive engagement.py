#!/usr/bin/env python
# coding: utf-8

# # ML CLASSIFIERS FOR COGNITIVE ENGAGEMENT PREDICTION

# In[ ]:


## Start with loading libraries and and splitting the data for train and test set


# In[1]:


##Libraries needed 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import randint
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC


###for plots
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus


# In[2]:


import pandas as pd
df = pd.read_csv('data_2.csv')
print(df.head())
print(df.shape)
print(df.dtypes)


# In[3]:


df["IsReply"] = df["IsReply"].astype("category")
print(df.dtypes)


# In[4]:


###let's check whether there are any missing values
col_mask=df.isnull().any(axis=0) 
row_mask=df.isnull().any(axis=1)
df.loc[row_mask,col_mask]


# In[5]:


#splitting the data in terms of features and target array 
#Splitting the data into independent and dependent variables
X = df.iloc[:,1:108].values
y = df.iloc[:,108].values
print('The independent features set: ')
print(X[:108,:])
print('The dependent variable: ')
print(y[:108])


# In[6]:


# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 21)


# ## Decision Tree 

# In[7]:


# Create Decision Tree classifer object
dt = DecisionTreeClassifier(random_state = 42)

# Train Decision Tree Classifer
dt_fit = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


print(confusion_matrix(y_test, y_pred))
print("Cohen's Kappa:",cohen_kappa_score(y_test, y_pred))


# In[8]:


print(dt.get_params())


# In[69]:


params_dt={'max_depth': [3, 6, 9, 12, None], 'min_samples_leaf':randint(1,10), 'max_features':randint(1,109), 'criterion':['gini', 'entropy']}


# In[70]:


random_dt=RandomizedSearchCV(dt, params_dt, n_jobs = -1, cv=10,scoring= 'accuracy')
random_dt.fit(X_train, y_train)


# In[11]:


print("Tuned Decision Tree Parameters: {}".format(random_dt.best_params_))
print("Best score is {}".format(random_dt.best_score_))
###the accuracy increases to .59 if min samples leaf = 20, max features = 50


# In[73]:


def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))
    print("Cohen's Kappa:",cohen_kappa_score(y_test, dt_classifier.predict(X_test)))
    print('Classification report:', classification_report(y_test, dt_classifier.predict(X_test)))
    print('Precision_recall_fscore_macro:',precision_recall_fscore_support(y_test, dt_classifier.predict(X_test), average='macro'))
    print('Precision_recall_fscore_micro:',precision_recall_fscore_support(y_test, dt_classifier.predict(X_test), average='micro'))
    print('Precision_recall_fscore_weighted:',precision_recall_fscore_support(y_test, dt_classifier.predict(X_test), average='weighted'))
    

print(random_dt.best_estimator_)
dt_best = random_dt.best_estimator_
evaluate_model(dt_best)


# In[84]:


dt = DecisionTreeClassifier(max_depth=6, max_features=35, min_samples_leaf=6,
                       random_state=42)

# Use the random grid to search for best hyperparameters
# First create the base model to tune

# Random search of parameters, using 10 fold cross validation, 
# search across 100 different combinations, and use all available cores

# Fit the random search model
dt.fit(X_train, y_train)
print(confusion_matrix(y_test, dt.predict(X_test)))
conf_mx=confusion_matrix(y_test, dt.predict(X_test))


# In[86]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[87]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[13]:


feature_names = df.columns.tolist()
features=feature_names[1:108]

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dt_best, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['0','1', '2', '3'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dt.png')
Image(graph.create_png())


# ## Random Forest

# In[14]:


rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[15]:


# Predicting the Test set results
y_pred = rf.predict(X_test)
#Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Eng'], colnames=['Predicted Eng']))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Cohen's Kappa:",cohen_kappa_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[16]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
criterion = ['gini', 'entropy']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features':max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion': criterion,
               
               
         }
print(random_grid)


# In[17]:


rf = RandomForestClassifier(random_state = 42)

# Use the random grid to search for best hyperparameters
# First create the base model to tune

# Random search of parameters, using 10 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 10, verbose=2, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[51]:


def evaluate_model(rf_classifier):
    print("Train Accuracy :", accuracy_score(y_train, rf_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, rf_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, rf_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, rf_classifier.predict(X_test)))
    print("Cohen's Kappa:",cohen_kappa_score(y_test, rf_classifier.predict(X_test)))
    print('Classification report:', classification_report(y_test, rf_classifier.predict(X_test)))
    print('Precision_recall_fscore_macro:',precision_recall_fscore_support(y_test, rf_classifier.predict(X_test), average='macro'))
    print('Precision_recall_fscore_micro:',precision_recall_fscore_support(y_test, rf_classifier.predict(X_test), average='micro'))
    print('Precision_recall_fscore_weighted:',precision_recall_fscore_support(y_test, rf_classifier.predict(X_test), average='weighted'))
    
    
    
    
print(rf_random.best_estimator_)
rf_best =rf_random.best_estimator_
evaluate_model(rf_best)


# In[88]:


rf = RandomForestClassifier(max_depth=60, min_samples_leaf=2, n_estimators=1600,
                       random_state=42)

# Use the random grid to search for best hyperparameters
# First create the base model to tune

# Random search of parameters, using 10 fold cross validation, 
# search across 100 different combinations, and use all available cores

# Fit the random search model
rf.fit(X_train, y_train)


# In[89]:


print(confusion_matrix(y_test, rf.predict(X_test)))
conf_mx=confusion_matrix(y_test, rf.predict(X_test))


# In[90]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[91]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[32]:


feature_names = df.columns.tolist()
features = feature_names[1:108]
print(len(features))


# In[34]:


import pandas as pd
feature_imp = pd.Series(rf.feature_importances_,index=features).sort_values(ascending=False)
feature_imp


# In[43]:


feature_imp[0:19]


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp[0:19], y=feature_imp.index[0:19])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[50]:


plt.savefig("feature_imp.png")


# ## Support Vector Machine

# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

# define model and parameters
model = SVC()
kernel = ['linear','poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
random = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
random_search = RandomizedSearchCV(model, random, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
random_result = random_search.fit(X, y)


# In[ ]:


def evaluate_model(svm_classifier):
    print("Train Accuracy :", accuracy_score(y_train, svm_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, svm_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, svm_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, svm_classifier.predict(X_test)))
    print("Cohen's Kappa:",cohen_kappa_score(y_test, svm_classifier.predict(X_test)))
    print('Classification report:', classification_report(y_test, svm_classifier.predict(X_test)))
    print('Precision_recall_fscore_macro:',precision_recall_fscore_support(y_test, svm_classifier.predict(X_test), average='macro'))
    print('Precision_recall_fscore_micro:',precision_recall_fscore_support(y_test, svm_classifier.predict(X_test), average='micro'))
    print('Precision_recall_fscore_weighted:',precision_recall_fscore_support(y_test, svm_classifier.predict(X_test), average='weighted'))
    
    
    
    
    
    
print(random_result.best_estimator_)
svm_best =random_result.best_estimator_
evaluate_model(svm_best)


# In[66]:


svm_clf = SVC(kernel = 'linear', C= 50)
svm_clf.fit(X_train, y_train)
svm_clf.predict(X_test)
svm_clf.score(X_test, y_test)


# In[92]:


print(confusion_matrix(y_test, svm_clf.predict(X_test)))
conf_mx=confusion_matrix(y_test, svm_clf.predict(X_test))


# In[93]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[94]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

