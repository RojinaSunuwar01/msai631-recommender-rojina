import pickle, numpy as np, pandas as pd
from scipy.sparse import csr_matrix

with open('models/cosine_model.pkl','rb') as f:
    M = pickle.load(f)

SIM      = M["sim_sparse"]
movie_ids= M["movie_ids"]
mid_map  = M["mid_map"]
UI       = M["UI"]
movies   = M["movies"]

def _user_index(user_id:int):
    try:
        return int(np.where(M["user_ids"] == user_id)[0][0])
    except IndexError:
        raise ValueError(f"Unknown or filtered-out user_id {user_id}")

def top_n_for_user(user_id:int, n:int=10):
    uidx = _user_index(user_id)
    user_row = UI.getrow(uidx)
    scores = user_row.dot(SIM).toarray().ravel()
    scores[user_row.indices] = -np.inf
    top_idx = np.argpartition(-scores, range(min(n, len(scores))))[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    top_mids = movie_ids[top_idx]
    out = movies[movies['movieId'].isin(top_mids)].copy()
    out['pred'] = out['movieId'].map({mid: float(scores[mid_map[mid]]) for mid in top_mids})
    return out.sort_values('pred', ascending=False).reset_index(drop=True)

def explain_reasons(user_id:int, movie_id:int, k:int=3):
    if movie_id not in mid_map:
        return []
    j = mid_map[movie_id]
    sim_j = SIM.getcol(j).toarray().ravel()
    uidx = _user_index(user_id)
    user_row = UI.getrow(uidx)
    liked_cols = user_row.indices[user_row.data >= 4.0]
    if liked_cols.size == 0:
        return []
    liked_scores = [(c, sim_j[c]) for c in liked_cols]
    liked_scores.sort(key=lambda x: x[1], reverse=True)
    liked_scores = liked_scores[:k]
    inv_mid = {v:k for k,v in mid_map.items()}
    titles = movies.set_index('movieId')['title']
    return [(titles.get(inv_mid[c], str(inv_mid[c])), float(UI[uidx, c])) for c, _ in liked_scores]
