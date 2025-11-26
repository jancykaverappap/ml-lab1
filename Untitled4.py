#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile

with zipfile.ZipFile("iris.zip", "r") as z:
    z.extractall(".")


# In[2]:


import zipfile
import os

zip_path = "iris.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("iris_extracted")

print("Extracted files:", os.listdir("iris_extracted"))


# In[3]:


import pandas as pd

folder = "iris_extracted"
csv_file = [f for f in os.listdir(folder) if f.endswith(".csv")][0]

df = pd.read_csv(os.path.join(folder, csv_file))
df.head()


# In[4]:


import os

folder = "iris_extracted"
print(os.listdir(folder))


# In[5]:


import pandas as pd
import os

# Path to extracted folder
folder = "iris_extracted"

# Load the .data file
file_path = os.path.join(folder, "iris.data")

df = pd.read_csv(file_path, header=None)

# Assign correct column names
df.columns = [
    'sepal_length', 'sepal_width',
    'petal_length', 'petal_width',
    'species'
]

df.head()


# In[6]:


df_numeric = df.drop(columns=['species'])
df_numeric.head()


# In[7]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)


# In[8]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters
df.head()


# In[9]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.title("K-Means Clustering on IRIS Dataset")
plt.show()


# In[10]:


inertia = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


# In[ ]:




