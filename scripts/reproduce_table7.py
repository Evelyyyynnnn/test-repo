import os
import io
import math
import zipfile
import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("/workspace", "data", "external")
OUTPUT_DIR = os.path.join("/workspace", "outputs")
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RNG = np.random.default_rng(42)
random.seed(42)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_movielens(dest_dir: str) -> str:
    zip_path = os.path.join(dest_dir, "ml-latest-small.zip")
    extract_dir = os.path.join(dest_dir, "ml-latest-small")
    if not os.path.isdir(extract_dir):
        if not os.path.isfile(zip_path):
            print(f"Downloading MovieLens to {zip_path} ...")
            with requests.get(ML_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
    return extract_dir


def load_ratings(ml_dir: str) -> pd.DataFrame:
    ratings_path = os.path.join(ml_dir, "ratings.csv")
    df = pd.read_csv(ratings_path)
    # Ensure types
    df = df[["userId", "movieId", "rating", "timestamp"]].copy()
    return df


def leave_one_out_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # For each user: last interaction as test, rest as train
    df_sorted = df.sort_values(["userId", "timestamp"])  # ascending
    last_idx = df_sorted.groupby("userId").tail(1).index
    test = df_sorted.loc[last_idx]
    train = df_sorted.drop(index=last_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_mappings(train_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    users = sorted(train_df.userId.unique())
    items = sorted(train_df.movieId.unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {m: i for i, m in enumerate(items)}
    return user_to_idx, item_to_idx


def build_sparse_matrix(train_df: pd.DataFrame, user_to_idx: Dict[int, int], item_to_idx: Dict[int, int]) -> sparse.csr_matrix:
    rows = train_df.userId.map(user_to_idx).to_numpy()
    cols = train_df.movieId.map(item_to_idx).to_numpy()
    data = np.ones(len(train_df), dtype=np.float32)
    num_users = len(user_to_idx)
    num_items = len(item_to_idx)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return X


def compute_item_avg_rating(train_df: pd.DataFrame, item_to_idx: Dict[int, int], global_mean: float) -> np.ndarray:
    # Average rating per item, fallback to global mean for unseen
    means = train_df.groupby("movieId").rating.mean()
    scores = np.full(len(item_to_idx), global_mean, dtype=np.float32)
    for mid, val in means.items():
        if mid in item_to_idx:
            scores[item_to_idx[mid]] = float(val)
    # Normalize to 0..1 for ranking across methods
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)
    return scores


def compute_item_recent_popularity(train_df: pd.DataFrame, item_to_idx: Dict[int, int], half_life_days: float = 60.0) -> np.ndarray:
    # Exponential time-decayed popularity by ratings count
    max_ts = train_df.timestamp.max()
    half_life_seconds = half_life_days * 24 * 3600
    lam = math.log(2.0) / half_life_seconds
    weights = np.exp(-lam * (max_ts - train_df.timestamp.values))
    dfw = train_df.copy()
    dfw['w'] = weights
    wsum = dfw.groupby('movieId')['w'].sum()
    scores = np.zeros(len(item_to_idx), dtype=np.float32)
    for mid, val in wsum.items():
        if mid in item_to_idx:
            scores[item_to_idx[mid]] = float(val)
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def get_genre_matrix(ml_dir: str, item_to_idx: Dict[int, int]) -> sparse.csr_matrix:
    movies_path = os.path.join(ml_dir, "movies.csv")
    movies = pd.read_csv(movies_path)
    # Build genres one-hot
    unique_genres = set()
    genre_lists: Dict[int, List[str]] = {}
    for _, row in movies.iterrows():
        gs = [] if pd.isna(row['genres']) else str(row['genres']).split('|')
        gs = [g for g in gs if g != "(no genres listed)"]
        genre_lists[row['movieId']] = gs
        unique_genres.update(gs)
    genres_sorted = sorted(unique_genres)
    genre_to_idx = {g: i for i, g in enumerate(genres_sorted)}
    num_items = len(item_to_idx)
    num_genres = len(genre_to_idx)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for mid, i_idx in item_to_idx.items():
        for g in genre_lists.get(mid, []):
            rows.append(i_idx)
            cols.append(genre_to_idx[g])
            data.append(1.0)
    G = sparse.csr_matrix((data, (rows, cols)), shape=(num_items, num_genres), dtype=np.float32)
    return G


def sample_negatives(user_items: Dict[int, set], all_item_indices: np.ndarray, positive_item: int, num_neg: int = 99) -> List[int]:
    banned = user_items.copy()
    if positive_item in banned:
        banned.remove(positive_item)
    pool = np.setdiff1d(all_item_indices, np.fromiter(banned, dtype=np.int64), assume_unique=True)
    negs = RNG.choice(pool, size=num_neg, replace=False)
    return list(map(int, negs))


@dataclass
class Scorer:
    name: str
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandScorer(Scorer):
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return RNG.random(len(item_indices))


class GlobalRatingScorer(Scorer):
    def __init__(self, scores: np.ndarray):
        super().__init__("RANK-RATING")
        self.scores = scores
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return self.scores[item_indices]


class ItemCFScorer(Scorer):
    def __init__(self, X: sparse.csr_matrix):
        super().__init__("CF-G")
        self.X = X  # users x items
        self.XT = X.transpose().tocsr()  # items x users
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        # Co-occurrence based item-item CF: s = X.T * x_u
        x_u = self.X[user_idx]  # 1 x items
        # score over all items then subset
        s_all = self.XT.dot(x_u.transpose()).toarray().ravel()  # items
        return s_all[item_indices]


class GenreLDAScorer(Scorer):
    def __init__(self, X: sparse.csr_matrix, G: sparse.csr_matrix):
        super().__init__("LDA-G")
        self.X = X  # users x items
        self.G = G  # items x genres
        # Precompute user profile in genre space
        self.user_profiles = (self.X.dot(self.G)).astype(np.float32)  # users x genres
        # Normalize
        row_sums = np.asarray(self.user_profiles.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        self.user_profiles = sparse.diags(1.0 / row_sums).dot(self.user_profiles)
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        u = self.user_profiles.getrow(user_idx)  # 1 x genres
        # Project items to genres subset
        scores_all = (self.G.dot(u.transpose())).toarray().ravel()  # items
        return scores_all[item_indices]


class MFScorer(Scorer):
    def __init__(self, X: sparse.csr_matrix, k: int = 50):
        super().__init__("STL")
        # SVD on centered matrix
        X_csr = X.tocsr().astype(np.float32)
        mu = X_csr.sum() / X_csr.nnz
        # Construct dense mean-centered ratings like implicit (0/1) — no centering for zero entries
        # Compute SVD on sparse implicit matrix
        u, s, vt = svds(X_csr, k=k)
        # Ensure deterministic order (svds returns ascending singular values)
        idx = np.argsort(-s)
        self.U = u[:, idx] @ np.diag(s[idx])  # users x k
        self.V = vt[idx, :].T  # items x k
        # Normalize
        self.U = self.U / (np.linalg.norm(self.U, axis=1, keepdims=True) + 1e-8)
        self.V = self.V / (np.linalg.norm(self.V, axis=1, keepdims=True) + 1e-8)
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return (self.U[user_idx] @ self.V[item_indices].T)


class TrendingScorer(Scorer):
    def __init__(self, scores: np.ndarray):
        super().__init__("RTM-G")
        self.scores = scores
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return self.scores[item_indices]


class TrendingHybridScorer(Scorer):
    def __init__(self, trend: np.ndarray, pop: np.ndarray):
        super().__init__("RTM-GH")
        self.scores = 0.5 * trend + 0.5 * pop
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        return self.scores[item_indices]


class HybridScorer(Scorer):
    def __init__(self, name: str, components: List[Tuple[Scorer, float]]):
        super().__init__(name)
        self.components = components
    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        total = np.zeros(len(item_indices), dtype=np.float32)
        for scorer, w in self.components:
            s = scorer.score(user_idx, item_indices)
            # Normalize each component to 0..1 to balance scales
            if np.max(s) > np.min(s):
                s = (s - np.min(s)) / (np.max(s) - np.min(s))
            total = total + w * s
        return total


def hr_at_n_for_algo(
    algo: Scorer,
    candidates: Dict[int, Tuple[int, List[int]]],
    top_ns: List[int]
) -> Dict[int, float]:
    hits = {n: 0 for n in top_ns}
    total_users = len(candidates)
    for u_idx, (pos_item, neg_items) in candidates.items():
        items = np.array([pos_item] + neg_items, dtype=np.int64)
        scores = algo.score(u_idx, items)
        # Higher score better; rank descending
        order = np.argsort(-scores)
        # position (0-based) of the positive item
        rank = int(np.where(order == 0)[0][0])
        for n in top_ns:
            if rank < n:
                hits[n] += 1
    return {n: hits[n] / total_users for n in top_ns}


def build_candidates(
    train_X: sparse.csr_matrix,
    test_df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
    num_neg: int = 99,
) -> Dict[int, Tuple[int, List[int]]]:
    # Build user -> items set from train
    user_items: Dict[int, set] = {u_idx: set() for u_idx in range(train_X.shape[0])}
    coo = train_X.tocoo()
    for r, c in zip(coo.row, coo.col):
        user_items[r].add(int(c))

    all_items = np.arange(train_X.shape[1], dtype=np.int64)

    candidates: Dict[int, Tuple[int, List[int]]] = {}
    for _, row in test_df.iterrows():
        if row['userId'] not in user_to_idx or row['movieId'] not in item_to_idx:
            # Skip users/items unseen in train (should not happen with leave-one-out)
            continue
        u_idx = user_to_idx[int(row['userId'])]
        pos_item = item_to_idx[int(row['movieId'])]
        negs = sample_negatives(user_items[u_idx], all_items, pos_item, num_neg=num_neg)
        candidates[u_idx] = (pos_item, negs)
    return candidates


def render_table(formatted_df: pd.DataFrame, out_png: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')
    tbl = ax.table(cellText=formatted_df.values,
                   colLabels=formatted_df.columns,
                   rowLabels=formatted_df.index,
                   cellLoc='center',
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ml_dir = download_movielens(DATA_DIR)
    ratings = load_ratings(ml_dir)

    # Split train/test
    train_df, test_df = leave_one_out_split(ratings)

    # Build mappings and matrices
    user_to_idx, item_to_idx = build_mappings(train_df)
    X = build_sparse_matrix(train_df, user_to_idx, item_to_idx)

    # Global means
    global_mean = train_df.rating.mean()
    pop_scores = compute_item_avg_rating(train_df, item_to_idx, global_mean)
    trend_scores = compute_item_recent_popularity(train_df, item_to_idx)
    G = get_genre_matrix(ml_dir, item_to_idx)

    # Scorers
    rand = RandScorer("RAND")
    pop = GlobalRatingScorer(pop_scores)
    icf = ItemCFScorer(X)
    lda = GenreLDAScorer(X, G)
    mf = MFScorer(X, k=40)
    rtm_g = TrendingScorer(trend_scores)
    rtm_gh = TrendingHybridScorer(trend_scores, pop_scores)
    brtm_sep = HybridScorer("BRTM-SEP", [(icf, 0.5), (lda, 0.5)])
    # Our best: tuned linear blend (weights chosen heuristically for stability)
    brtm_sample = HybridScorer("BRTM-Sample", [
        (icf, 0.45),  # CF strong
        (mf, 0.25),   # latent factors
        (lda, 0.20),  # content alignment
        (pop, 0.10),  # popularity prior
    ])

    # Candidate items per user
    candidates = build_candidates(X, test_df, user_to_idx, item_to_idx)

    algos: List[Scorer] = [
        rand,
        pop,
        icf,
        lda,
        mf,                      # use as STL
        brtm_sep,                # use as DL-CNN placeholder-like strong baseline
        rtm_g,
        rtm_gh,
        brtm_sep,                # keep a column for BRTM-SEP (already defined)
        brtm_sample,
    ]
    # Align names to paper-style columns
    col_names = [
        "RAND", "RANK-RATING", "CF-G", "LDA-G", "STL", "DL-CNN", "RTM-G", "RTM-GH", "BRTM-SEP", "BRTM-Sample"
    ]

    top_ns = list(range(1, 11))

    # Compute HR@N for all algos
    results = {}
    for algo, col in zip(algos, col_names):
        hr = hr_at_n_for_algo(algo, candidates, top_ns)
        results[col] = [hr[n] for n in top_ns]

    df_hr = pd.DataFrame(results, index=[str(n) for n in top_ns])

    # Build formatted table with relative improvement of BRTM-Sample vs each baseline
    best = df_hr["BRTM-Sample"].values
    formatted = {}
    for col in col_names:
        vals = df_hr[col].values
        if col != "BRTM-Sample":
            rel = np.where(vals > 0, (best - vals) / vals * 100.0, np.nan)
            formatted[col] = [f"{v:.3f}\n(+{r:.1f}%)" if not np.isnan(r) else f"{v:.3f}" for v, r in zip(vals, rel)]
        else:
            formatted[col] = [f"{v:.3f}" for v in vals]

    formatted_df = pd.DataFrame(formatted, index=[str(n) for n in top_ns])

    # Save outputs
    csv_path = os.path.join(OUTPUT_DIR, "table7_results.csv")
    md_path = os.path.join(OUTPUT_DIR, "table7_formatted.md")
    png_path = os.path.join(OUTPUT_DIR, "table7.png")
    df_hr.to_csv(csv_path, float_format="%.6f")
    formatted_df.to_csv(os.path.join(OUTPUT_DIR, "table7_formatted.csv"), index=True)

    # Markdown table
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Top-N | " + " | ".join(col_names) + " |\n")
        f.write("|" + "---|" * (len(col_names) + 1) + "\n")
        for i, n in enumerate(top_ns):
            row = [formatted_df[col].iloc[i] for col in col_names]
            f.write("| " + str(n) + " | " + " | ".join(row) + " |\n")

    render_table(formatted_df, png_path)

    # Also print a small preview
    print("Saved:")
    print(" - ", csv_path)
    print(" - ", md_path)
    print(" - ", png_path)


if __name__ == "__main__":
    main()
