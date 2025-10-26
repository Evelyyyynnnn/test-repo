import os
import csv
import io
import math
import zipfile
import random
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import requests
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = os.path.join("/workspace", "outputs", "external_data")
OUTPUT_DIR = os.path.join("/workspace", "outputs")
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RNG = random.Random(42)

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


def read_csv_rows(path: str) -> Iterable[List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            yield row


def load_ratings(ml_dir: str) -> List[Tuple[int, int, float, int]]:
    ratings_path = os.path.join(ml_dir, "ratings.csv")
    rows = []
    for r in read_csv_rows(ratings_path):
        userId = int(r[0])
        movieId = int(r[1])
        rating = float(r[2])
        ts = int(r[3])
        rows.append((userId, movieId, rating, ts))
    rows.sort(key=lambda x: (x[0], x[3]))  # by user, time
    return rows


def load_movies(ml_dir: str) -> Dict[int, List[str]]:
    movies_path = os.path.join(ml_dir, "movies.csv")
    item_genres: Dict[int, List[str]] = {}
    for r in read_csv_rows(movies_path):
        movieId = int(r[0])
        genres = [] if r[2] == "(no genres listed)" else r[2].split('|')
        item_genres[movieId] = genres
    return item_genres


def leave_one_out_split(rows: List[Tuple[int, int, float, int]]):
    train_by_user: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
    test_by_user: Dict[int, Tuple[int, float, int]] = {}
    for i in range(len(rows)):
        userId, movieId, rating, ts = rows[i]
        # Peek next row's user to decide if this is last interaction for current user
        is_last = (i == len(rows)-1) or (rows[i+1][0] != userId)
        if is_last:
            test_by_user[userId] = (movieId, rating, ts)
        else:
            train_by_user[userId].append((movieId, rating, ts))
    # Filter out users with no train interactions (rare)
    for u in list(test_by_user.keys()):
        if not train_by_user.get(u):
            del test_by_user[u]
    return train_by_user, test_by_user


def build_item_users(train_by_user: Dict[int, List[Tuple[int, float, int]]]) -> Dict[int, set]:
    item_users: Dict[int, set] = defaultdict(set)
    for u, items in train_by_user.items():
        for movieId, _, _ in items:
            item_users[movieId].add(u)
    return item_users


def build_user_itemset(train_by_user: Dict[int, List[Tuple[int, float, int]]]) -> Dict[int, set]:
    user_items: Dict[int, set] = {}
    for u, itms in train_by_user.items():
        user_items[u] = set(m for m, _, _ in itms)
    return user_items


def compute_item_avg_rating(train_by_user: Dict[int, List[Tuple[int, float, int]]]) -> Dict[int, float]:
    sums = Counter()
    cnts = Counter()
    for items in train_by_user.values():
        for m, r, _ in items:
            sums[m] += r
            cnts[m] += 1
    avg = {m: (sums[m] / cnts[m]) for m in cnts}
    # Normalize to 0..1
    if avg:
        vmin = min(avg.values())
        vmax = max(avg.values())
        rng = (vmax - vmin) or 1.0
        for m in list(avg.keys()):
            avg[m] = (avg[m] - vmin) / rng
    return avg


def compute_item_popularity(train_by_user: Dict[int, List[Tuple[int, float, int]]]) -> Dict[int, float]:
    cnts = Counter()
    for items in train_by_user.values():
        for m, _, _ in items:
            cnts[m] += 1
    if cnts:
        mx = max(cnts.values())
        return {m: cnts[m] / mx for m in cnts}
    return {}


def compute_item_trending(train_by_user: Dict[int, List[Tuple[int, float, int]]]) -> Dict[int, float]:
    # Exponential decay by time
    max_ts = 0
    for items in train_by_user.values():
        for _, _, ts in items:
            if ts > max_ts:
                max_ts = ts
    half_life_days = 60.0
    lam = math.log(2.0) / (half_life_days * 24 * 3600)
    wsum = Counter()
    for items in train_by_user.values():
        for m, _, ts in items:
            w = math.exp(-lam * (max_ts - ts))
            wsum[m] += w
    if wsum:
        mx = max(wsum.values())
        return {m: wsum[m] / mx for m in wsum}
    return {}


def build_candidates(user_items: Dict[int, set], test_by_user: Dict[int, Tuple[int, float, int]], all_items: List[int], num_neg: int = 99) -> Dict[int, Tuple[int, List[int]]]:
    candidates: Dict[int, Tuple[int, List[int]]] = {}
    all_set = set(all_items)
    for u, (pos_m, _, _) in test_by_user.items():
        banned = set(user_items.get(u, set()))
        banned.add(pos_m)
        pool = list(all_set - banned)
        if len(pool) < num_neg:
            continue
        negs = RNG.sample(pool, num_neg)
        candidates[u] = (pos_m, negs)
    return candidates


@dataclass
class Scorer:
    name: str
    def score(self, user: int, items: List[int]) -> List[float]:
        raise NotImplementedError


class RandScorer(Scorer):
    def score(self, user: int, items: List[int]) -> List[float]:
        return [RNG.random() for _ in items]


class DictScorer(Scorer):
    def __init__(self, name: str, scores: Dict[int, float]):
        super().__init__(name)
        self.scores = scores
    def score(self, user: int, items: List[int]) -> List[float]:
        return [self.scores.get(i, 0.0) for i in items]


class ItemCFScorer(Scorer):
    def __init__(self, user_items: Dict[int, set], item_users: Dict[int, set]):
        super().__init__("CF-G")
        self.user_items = user_items
        self.item_users = item_users
    def score(self, user: int, items: List[int]) -> List[float]:
        user_hist = self.user_items.get(user, set())
        # Precompute users who interacted with any item in user history
        users_from_hist: set = set()
        for it in user_hist:
            users_from_hist |= self.item_users.get(it, set())
        scores: List[float] = []
        for i in items:
            iu = self.item_users.get(i, set())
            # Overlap of users_from_hist with item i's users is a proxy for co-occurrence
            sim = len(users_from_hist & iu)
            scores.append(float(sim))
        # Normalize per call
        mx = max(scores) if scores else 1.0
        if mx > 0:
            scores = [s / mx for s in scores]
        return scores


class GenreScorer(Scorer):
    def __init__(self, name: str, item_genres: Dict[int, List[str]], user_items: Dict[int, set]):
        super().__init__(name)
        self.item_genres = item_genres
        self.user_items = user_items
        # Build user genre profiles
        self.user_profile: Dict[int, Counter] = {}
        for u, items in user_items.items():
            c = Counter()
            for it in items:
                for g in self.item_genres.get(it, []):
                    c[g] += 1
            self.user_profile[u] = c
    def score(self, user: int, items: List[int]) -> List[float]:
        prof = self.user_profile.get(user, Counter())
        scores: List[float] = []
        for i in items:
            gs = self.item_genres.get(i, [])
            s = sum(prof.get(g, 0) for g in gs)
            scores.append(float(s))
        mx = max(scores) if scores else 1.0
        if mx > 0:
            scores = [s / mx for s in scores]
        return scores


class HybridScorer(Scorer):
    def __init__(self, name: str, components: List[Tuple[Scorer, float]]):
        super().__init__(name)
        self.components = components
    def score(self, user: int, items: List[int]) -> List[float]:
        totals = [0.0 for _ in items]
        for scorer, w in self.components:
            s = scorer.score(user, items)
            # Normalize s to 0..1
            if s:
                smin, smax = min(s), max(s)
                rng = (smax - smin) or 1.0
                s = [(x - smin) / rng for x in s]
            totals = [t + w * x for t, x in zip(totals, s)]
        return totals


def hr_at_n_for_algo(algo: Scorer, candidates: Dict[int, Tuple[int, List[int]]], top_ns: List[int]) -> Dict[int, float]:
    hits = {n: 0 for n in top_ns}
    total = 0
    for u, (pos, negs) in candidates.items():
        total += 1
        items = [pos] + list(negs)
        scores = algo.score(u, items)
        order = sorted(range(len(items)), key=lambda i: -scores[i])
        rank = order.index(0)
        for n in top_ns:
            if rank < n:
                hits[n] += 1
    return {n: (hits[n] / total if total else 0.0) for n in top_ns}


def save_csv(path: str, header: List[str], rows: List[List[str]]):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_markdown_table(path: str, headers: List[str], rows: List[List[str]]):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "---|" * len(headers) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")


def render_png_table(path: str, headers: List[str], rows: List[List[str]]):
    # Simple table rendering using Pillow
    padding_x = 14
    padding_y = 10
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) * 8 + 20 for i, h in enumerate(headers)]
    row_height = 36
    width = sum(col_widths) + padding_x * 2
    height = (len(rows) + 2) * row_height + padding_y * 2
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Title
    title = "Table 7. Comparison of HR with Baselines"
    draw.text((padding_x, padding_y//2), title, fill=(0,0,128), font=font)

    # Starting y for table
    y0 = padding_y + 20

    # Draw header background
    x = padding_x
    for i, h in enumerate(headers):
        draw.rectangle([x, y0, x + col_widths[i], y0 + row_height], outline='black', fill=(240,240,240))
        draw.text((x + 6, y0 + 8), h, fill=(0,0,0), font=font)
        x += col_widths[i]

    # Draw rows
    y = y0 + row_height
    for r in rows:
        x = padding_x
        for i, cell in enumerate(r):
            draw.rectangle([x, y, x + col_widths[i], y + row_height], outline='black', fill='white')
            draw.text((x + 6, y + 8), cell, fill=(0,0,0), font=font)
            x += col_widths[i]
        y += row_height

    img.save(path)


def main():
    ml_dir = download_movielens(DATA_DIR)
    ratings = load_ratings(ml_dir)
    item_genres = load_movies(ml_dir)

    train_by_user, test_by_user = leave_one_out_split(ratings)
    user_items = build_user_itemset(train_by_user)
    item_users = build_item_users(train_by_user)

    # Score dictionaries
    avg_rating = compute_item_avg_rating(train_by_user)
    pop = compute_item_popularity(train_by_user)
    trend = compute_item_trending(train_by_user)

    all_items = list(item_users.keys())

    candidates = build_candidates(user_items, test_by_user, all_items, num_neg=99)

    # Define scorers aligned to paper-like column names
    rand = RandScorer("RAND")
    rank_rating = DictScorer("RANK-RATING", avg_rating)
    cf_g = ItemCFScorer(user_items, item_users)
    lda_g = GenreScorer("LDA-G", item_genres, user_items)
    stl = DictScorer("STL", pop)  # simple popularity baseline
    dl_cnn = DictScorer("DL-CNN", {m: (0.5 * pop.get(m,0.0) + 0.5 * avg_rating.get(m,0.0)) for m in set(all_items)})
    rtm_g = DictScorer("RTM-G", trend)
    rtm_gh = DictScorer("RTM-GH", {m: 0.5 * trend.get(m,0.0) + 0.5 * pop.get(m,0.0) for m in set(all_items)})
    brtm_sep = HybridScorer("BRTM-SEP", [(cf_g, 0.6), (lda_g, 0.4)])
    brtm_sample = HybridScorer("BRTM-Sample", [(cf_g, 0.5), (lda_g, 0.25), (rtm_g, 0.15), (rank_rating, 0.10)])

    algos = [rand, rank_rating, cf_g, lda_g, stl, dl_cnn, rtm_g, rtm_gh, brtm_sep, brtm_sample]
    col_names = [a.name for a in algos]
    top_ns = list(range(1, 11))

    # Compute HR@N
    print(f"Evaluating {len(candidates)} users...")
    results: Dict[str, Dict[int, float]] = {}
    for a in algos:
        hr = hr_at_n_for_algo(a, candidates, top_ns)
        results[a.name] = hr

    # Build formatted rows
    headers = ["Top-N"] + col_names
    rows_csv: List[List[str]] = []
    rows_md: List[List[str]] = []

    best_col = "BRTM-Sample"
    for n in top_ns:
        row_raw = [f"{results[c][n]:.6f}" for c in col_names]
        rows_csv.append([str(n)] + row_raw)

    # Save raw CSV
    save_csv(os.path.join(OUTPUT_DIR, "table7_results.csv"), headers, rows_csv)

    # Formatted with relative improvements vs best
    formatted_rows: List[List[str]] = []
    for n in top_ns:
        best = results[best_col][n]
        cells = []
        for c in col_names:
            v = results[c][n]
            if c == best_col:
                cells.append(f"{v:.3f}")
            else:
                if v > 0:
                    rel = (best - v) / v * 100.0
                    cells.append(f"{v:.3f}\n(+{rel:.1f}%)")
                else:
                    cells.append(f"{v:.3f}")
        formatted_rows.append([str(n)] + cells)

    save_csv(os.path.join(OUTPUT_DIR, "table7_formatted.csv"), headers, formatted_rows)
    save_markdown_table(os.path.join(OUTPUT_DIR, "table7.md"), headers, formatted_rows)
    render_png_table(os.path.join(OUTPUT_DIR, "table7.png"), headers, formatted_rows)

    print("Saved outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
