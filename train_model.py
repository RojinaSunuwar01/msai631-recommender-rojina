import os, pickle, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("models", exist_ok=True)

print("Loading data...")
ratings = pd.read_csv('data/ratings.csv')
movies  = pd.read_csv('data/movies.csv')

MIN_RATINGS = 20
print(f"Filtering movies with at least {MIN_RATINGS} ratings...")
movie_counts = ratings['movieId'].value_counts()
keep_movies = movie_counts[movie_counts >= MIN_RATINGS].index
ratings = ratings[ratings['movieId'].isin(keep_movies)]
movies  = movies[movies['movieId'].isin(keep_movies)]

print("Building user-item matrix...")
user_ids  = np.sort(ratings['userId'].unique())
movie_ids = np.sort(ratings['movieId'].unique())
uid_map = {u:i for i,u in enumerate(user_ids)}
mid_map = {m:i for i,m in enumerate(movie_ids)}

rows = ratings['userId'].map(uid_map).to_numpy()
cols = ratings['movieId'].map(mid_map).to_numpy()
data = ratings['rating'].to_numpy(dtype=np.float32)
UI = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

print("Mean-centering per user...")
UI_center = UI.tolil(copy=True)
for u in range(UI.shape[0]):
    nz = UI.getrow(u).data
    mu = float(np.mean(nz)) if len(nz) else 0.0
    if len(nz):
        UI_center.rows[u] = UI.getrow(u).indices.tolist()
        UI_center.data[u] = (UI.getrow(u).data - mu).tolist()
UI_center = UI_center.tocsr()

print("Computing item-item cosine similarities (sparse)...")
SIM = cosine_similarity(UI_center.T, dense_output=False)

TOPK = 50
print(f"Keeping top {TOPK} neighbors per item...")
SIM = SIM.tolil()
for j in range(SIM.shape[0]):
    row = SIM.rows[j]; dat = SIM.data[j]
    if len(dat) > TOPK:
        idx_sorted = np.argsort(dat)[::-1]
        keep = set(idx_sorted[:TOPK])
        SIM.rows[j] = [row[k] for k in keep]
        SIM.data[j] = [dat[k] for k in keep]
SIM = SIM.tocsr()

model = {
    "sim_sparse": SIM,
    "movie_ids": movie_ids,
    "mid_map": mid_map,
    "user_ids": user_ids,
    "UI": UI,
    "movies": movies[['movieId','title','genres']]
}

with open('models/cosine_model.pkl','wb') as f:
    pickle.dump(model, f)

print("✅ Model saved to models/cosine_model.pkl")
