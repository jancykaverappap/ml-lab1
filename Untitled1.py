#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np




# In[3]:


X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)
print("Dataset shape:", X.shape, y.shape)


# In[4]:


print("First 5 feature rows:\n", X[:5])
print("First 5 labels:", y[:5])



# In[6]:


import pandas as pd

# Assuming X is your features (numpy array) and y is labels (numpy array)

# Create a DataFrame for features with column names
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])

# Add the label column
df['label'] = y

# Display the first 5 rows (records)
print(df.head(5))


# In[7]:


import pandas as pd

# Assuming X and y are your features and labels
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
df['label'] = y

# Display last 5 records
print(df.tail(5))


# In[8]:


import pandas as pd
import numpy as np

# Example dataset
df = pd.DataFrame(np.random.randn(10, 5), columns=[f"feature_{i+1}" for i in range(5)])
df['label'] = np.random.randint(0, 3, size=10)

# Set pandas option to display all rows and columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

print(df)


# In[9]:


import pandas as pd

# Assuming X and y are your features and labels from the dataset creation step
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
df['label'] = y

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df)


# In[10]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate dummy dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# 2. Two-way split: Train/Test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Logistic Regression model
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 4. Predict and evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print(f"Test accuracy: {accuracy:.4f}")


# In[11]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate dummy dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# 2. First split: Train+Validation (80%), Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Second split: Train (75% of 80% = 60%), Validation (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Train size: {X_train.shape[0]}")        # ~3000 records
print(f"Validation size: {X_val.shape[0]}")     # ~1000 records
print(f"Test size: {X_test.shape[0]}")          # ~1000 records

# 4. Train logistic regression on train set
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 6. Optionally evaluate on test set after model tuning
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")


# In[12]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# 1. Create dummy dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# 2. Define K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Define logistic regression model
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)

# 4. Run cross-validation and get accuracy scores
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("K-Fold accuracies for each fold:", cv_scores)
print(f"Mean K-Fold accuracy: {cv_scores.mean():.4f}")


# In[13]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Create dummy dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# Define K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize Logistic Regression model
    model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on test fold
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print(f"Fold {fold} accuracy: {acc:.4f}")

print(f"Mean accuracy across folds: {np.mean(accuracies):.4f}")


# In[14]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Create dummy dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression model
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate F1 score (macro-average)
f1_macro = f1_score(y_test, y_pred, average='macro')

# Calculate F1 score (weighted-average)
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"Macro F1 score: {f1_macro:.4f}")
print(f"Weighted F1 score: {f1_weighted:.4f}")


# In[15]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Create dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train logistic regression model
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 4. Predict on test set
y_pred = model.predict(X_test)

# 5. Show some predictions vs true labels
for i in range(5):  # show first 5 predictions
    print(f"Sample {i+1}: Predicted = {y_pred[i]}, Actual = {y_test[i]}")


# In[17]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Generate dataset
X, y = make_classification(
    n_samples=5000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    random_state=42
)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train logistic regression
model = LogisticRegression(multi_class='multinomial', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 4. Predict on test data
y_pred = model.predict(X_test)

# 5. Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Optional: Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# In[ ]:




