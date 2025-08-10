import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from umap.umap_ import UMAP
from sklearn.multioutput import MultiOutputClassifier

n_number = 17971719
random.seed(n_number)

data = pd.read_csv('musicData.csv',delimiter=',')

data = data.replace('?', np.nan)
data = data.replace(-1, np.nan)
data_cleaned = data.dropna()

labels = data_cleaned['music_genre']

df_encoded = pd.get_dummies(
    data_cleaned[['key', 'mode', 'music_genre']],
    prefix=['key', 'mode', 'music_genre']
).astype(int)
data_cleaned_dropped = data_cleaned.drop(columns=['key', 'mode', 'music_genre'])
data_cleaned = pd.concat([data_cleaned_dropped, df_encoded], axis=1)

## Train/Test Split by Genre 

genre_columns = [col for col in data_cleaned.columns if col.startswith('music_genre_')]
train_list = []
test_list = []

for genre_col in genre_columns: 
    g = data_cleaned[data_cleaned[genre_col] == 1]
    g_train, g_test = train_test_split(
        g,
        test_size=0.1,
        random_state=n_number,
        shuffle=True
    )
    train_list.append(g_train)
    test_list.append(g_test)

train_data = pd.concat(train_list)
test_data = pd.concat(test_list)

X_train = train_data.drop(columns=genre_columns)
y_train = train_data[genre_columns].values

X_test = test_data.drop(columns=genre_columns)
y_test = test_data[genre_columns].values

print(X_test.columns.tolist())

numeric_cols = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']  # example
categorical_cols = data_cleaned.columns[15:29].tolist()

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
X_test_num_scaled = scaler.transform(X_test[numeric_cols])
X_train_cat = X_train[categorical_cols].values
X_test_cat = X_test[categorical_cols].values
X_train_scaled = np.concatenate([X_train_num_scaled, X_train_cat], axis=1)
X_test_scaled = np.concatenate([X_test_num_scaled, X_test_cat], axis=1)

# PERFORMING PCA

pca = PCA(n_components=6)  

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained_var = pca.explained_variance_ratio_
print(f'''The explained variance of PC1 is {explained_var[0]:.2%}
The explained variance of PC2 is {explained_var[1]:.2%}
The total explained variance from PC1, PC2, and PC3 is {explained_var[0] + explained_var[1] + explained_var[2]:.2%}''')

eigenvalues = pca.explained_variance_
num_eigen = (eigenvalues > 1).sum()
print('Number of eigenvalues > 1:', num_eigen)

plt.figure(figsize=(5, 5))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], s=10)   
plt.title('PCA Projection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component') 

# PERFORMING UMAP

umap = UMAP(n_components=6, init='random', random_state=n_number)

X_train_umap = umap.fit_transform(X_train_scaled)
X_test_umap = umap.transform(X_test_scaled)

plt.figure(figsize=(5, 5))
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], s=10)   
plt.title('umap Projection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component') 

genre_cols = [
    'music_genre_Electronic',
    'music_genre_Anime',
    'music_genre_Jazz',
    'music_genre_Alternative',
    'music_genre_Country',
    'music_genre_Rap',
    'music_genre_Blues',
    'music_genre_Rock',
    'music_genre_Classical',
    'music_genre_Hip-Hop'
]

# Plotting by genre

colors = plt.cm.tab10.colors  

plt.figure(figsize=(12, 8))

for i, genre_col in enumerate(genre_cols):
    genre_indices = np.where(y_test[:, i] == 1)[0]
    
    if len(genre_indices) > 0:
        plt.scatter(
            X_test_pca[genre_indices, 0],
            X_test_pca[genre_indices, 1],
            color=colors[i % len(colors)],
            s=30,
            alpha=0.7,
            label=genre_col.replace('music_genre_', '')
        )

plt.title('PCA 2D projection colored by genre')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


plt.figure(figsize=(12, 8))

for i, genre_col in enumerate(genre_cols):
    genre_indices = np.where(y_test[:, i] == 1)[0]
    
    if len(genre_indices) > 0:
        plt.scatter(
            X_test_umap[genre_indices, 0],
            X_test_umap[genre_indices, 1],
            color=colors[i % len(colors)],
            s=30,
            alpha=0.7,
            label=genre_col.replace('music_genre_', '')
        )

plt.title('umap 2D projection colored by genre')
plt.xlabel('umap Component 1')
plt.ylabel('umap Component 2')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# CLUSTERING STEP

kmeans = cluster.KMeans(n_clusters=10, n_init='auto', random_state=n_number)
kmeans.fit(X_train_pca)
labels = kmeans.predict(X_test_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=labels, s=50)
plt.title(f"PCA Projection w/ kMeans (k={10})")
plt.xlabel("PCA Dim. 1")
plt.ylabel("PCA Dim. 2")

kmeans = cluster.KMeans(n_clusters=10, n_init='auto', random_state=n_number)
kmeans.fit(X_train_umap)
labels = kmeans.predict(X_test_umap)

plt.figure(figsize=(8, 6))
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1], c=labels, s=50)
plt.title(f"PCA Projection w/ kMeans (k={10})")
plt.xlabel("PCA Dim. 1")
plt.ylabel("PCA Dim. 2")


# PREDICTION STEP -------------------------------------------------------------------------------------------------------------------------------

y_train = train_data[genre_columns].values
y_test = test_data[genre_columns].values

lr = LogisticRegression(max_iter=500)
multi_lr = MultiOutputClassifier(lr)
multi_lr.fit(X_train_pca, y_train)
y_pred = multi_lr.predict(X_test_pca)
print("Accuracy score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Get predicted probabilities for PCA
y_score = multi_lr.predict_proba(X_test_pca)
y_score = np.array([prob[:, 1] for prob in y_score]).T  

macro_roc_auc = roc_auc_score(y_test, y_score, average='macro')
micro_roc_auc = roc_auc_score(y_test, y_score, average='micro')

print(f"Macro AUC for PCA: {macro_roc_auc:.3f}")
print(f"Micro AUC for PCA: {micro_roc_auc:.3f}")

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))

for i in range(y_test.shape[1]):
    fpr, tpr, _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
    auc_score = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (PCA)')
plt.legend(loc='lower right')

# Get predicted probabilities for UMAP
lr = LogisticRegression(max_iter=500)
multi_lr = MultiOutputClassifier(lr)
multi_lr.fit(X_train_umap, y_train)
y_pred = multi_lr.predict(X_test_umap)

y_score = multi_lr.predict_proba(X_test_umap)
 
y_score = np.array([prob[:, 1] for prob in y_score]).T  

# Calculate the AUROC  
macro_roc_auc = roc_auc_score(y_test, y_score, average='macro')
micro_roc_auc = roc_auc_score(y_test, y_score, average='micro')

print(f"Macro AUC for UMAP: {macro_roc_auc:.3f}")
print(f"Micro AUC for UMAP: {micro_roc_auc:.3f}")

plt.figure(figsize=(10, 8))

for i in range(y_test.shape[1]):
    fpr, tpr, _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
    auc_score = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (UMAP)')
plt.legend(loc='lower right')

plt.show() 

