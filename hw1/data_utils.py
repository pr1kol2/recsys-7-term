"""Amazon Video Games: downloads, time splits, and item features."""
from pathlib import Path
import ast
import gzip
import hashlib
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = 'https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html'
BASE_URL = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
FILES = ('ratings_Video_Games.csv', 'meta_Video_Games.json.gz')
FILE_SIZES = {'ratings_Video_Games.csv': 53_949_503,
              'meta_Video_Games.json.gz': 23_557_720}
TRAIN_END = int(pd.Timestamp('2013-07-01', tz='UTC').timestamp())
VALID_END = int(pd.Timestamp('2014-01-01', tz='UTC').timestamp())
MAX_ITEMS = 4000
MIN_TRAIN_INTERACTIONS = 3


def download_data(directory='data/raw'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = directory / name
        if path.exists() and path.stat().st_size != FILE_SIZES[name]:
            raise ValueError(f'{path}: incomplete or changed source file; remove it and retry')
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            urllib.request.urlretrieve(BASE_URL + name, temporary)
            if temporary.stat().st_size != FILE_SIZES[name]:
                raise ValueError(f'{temporary}: incomplete or changed source file')
            temporary.replace(path)
    return directory


def file_hashes(directory):
    hashes = {}
    for name in FILES:
        digest = hashlib.sha256()
        with (Path(directory) / name).open('rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


def load_ratings(directory):
    raw = pd.read_csv(Path(directory) / FILES[0],
                      names=['user_id', 'item_id', 'rating', 'timestamp'],
                      dtype={'user_id': str, 'item_id': str})
    audit = {'raw_rows': len(raw), 'raw_users': raw.user_id.nunique(),
             'raw_items': raw.item_id.nunique(),
             'missing': raw.isna().sum().to_dict(),
             'duplicate_rows': int(raw.duplicated().sum()),
             'duplicate_pairs': int(raw.duplicated(['user_id', 'item_id']).sum())}
    clean = raw.dropna().copy()
    valid = clean.rating.between(1, 5) & (clean.timestamp > 0)
    audit['invalid_rows'] = int((~valid).sum())
    clean = clean.loc[valid].sort_values(
        ['timestamp', 'user_id', 'item_id'], kind='stable')
    # An item is seen from the user's first review onward.
    clean = clean.drop_duplicates(['user_id', 'item_id'], keep='first')
    clean.timestamp = clean.timestamp.astype('int64')
    audit['clean_rows'] = len(clean)
    return raw, clean, audit


def select_cohort(clean):
    past = clean.loc[clean.timestamp < TRAIN_END]
    counts = past.groupby('item_id').size().rename('n').reset_index()
    counts = counts.sort_values(['n', 'item_id'], ascending=[False, True])
    item_ids = sorted(counts.head(MAX_ITEMS).item_id)
    past = past.loc[past.item_id.isin(item_ids)]
    user_counts = past.groupby('user_id').size()
    user_ids = sorted(user_counts[user_counts >= MIN_TRAIN_INTERACTIONS].index)
    selected = clean.loc[clean.item_id.isin(item_ids) & clean.user_id.isin(user_ids)].copy()
    # Filtering users can leave some items without training ratings.
    item_ids = sorted(selected.loc[selected.timestamp < TRAIN_END, 'item_id'].unique())
    selected = selected.loc[selected.item_id.isin(item_ids)].copy()
    selected['user'] = pd.Categorical(selected.user_id, categories=user_ids).codes
    selected['item'] = pd.Categorical(selected.item_id, categories=item_ids).codes
    selected = selected.reset_index(drop=True)
    assert 100_000 <= len(selected) <= 2_000_000, len(selected)
    assert len(user_ids) >= 1000 and len(item_ids) >= 1000
    train = selected.loc[selected.timestamp < TRAIN_END].copy()
    valid = selected.loc[selected.timestamp.between(TRAIN_END, VALID_END, inclusive='left')].copy()
    test = selected.loc[selected.timestamp >= VALID_END].copy()
    return selected, train, valid, test, user_ids, item_ids


def interaction_matrix(frame, shape, column='rating'):
    values = np.ones(len(frame), dtype=np.float32) if column is None else frame[column].to_numpy()
    matrix = sparse.csr_matrix((values, (frame.user, frame.item)), shape=shape)
    matrix.sort_indices()
    return matrix


def positive_matrix(frame, shape):
    return interaction_matrix(frame.loc[frame.rating >= 4], shape, column=None)


def load_metadata(directory, item_ids):
    wanted = set(item_ids)
    records = {}
    raw_count = 0
    with gzip.open(Path(directory) / FILES[1], 'rt', encoding='utf-8') as stream:
        for line in stream:
            raw_count += 1
            # The file contains Python dictionaries, so json.loads cannot parse it.
            record = ast.literal_eval(line)
            item_id = record['asin']
            if item_id not in wanted or item_id in records:
                continue
            categories = sorted({category for path in record.get('categories', [])
                                 for category in path if category != 'Video Games'})
            records[item_id] = {'title': record.get('title') or '', 'categories': categories}
    rows = [dict(item_id=item_id, **records.get(item_id, {'title': '', 'categories': []}))
            for item_id in item_ids]
    return pd.DataFrame(rows), raw_count


def category_features(metadata):
    # Use categories from training items only; keep the vocabulary fixed for test.
    vocabulary = sorted({category for categories in metadata.categories for category in categories})
    vocabulary.append('__missing_category__')
    mapping = {name: i for i, name in enumerate(vocabulary)}
    rows, columns, weights = [], [], []
    for item, categories in enumerate(metadata.categories):
        categories = categories or ['__missing_category__']
        for category in categories:
            rows.append(item)
            columns.append(mapping[category])
            weights.append(1 / np.sqrt(len(categories)))
    matrix = sparse.csr_matrix((weights, (rows, columns)),
                               shape=(len(metadata), len(vocabulary)), dtype=np.float32)
    return matrix, vocabulary


def split_audit(clean, selected, train, valid, test, user_ids, item_ids):
    rows = []
    for name, low, high, part in [('train', 0, TRAIN_END, train),
                                  ('validation', TRAIN_END, VALID_END, valid),
                                  ('test', VALID_END, np.inf, test)]:
        original = clean.loc[(clean.timestamp >= low) & (clean.timestamp < high)]
        known_users = original.user_id.isin(user_ids)
        known_items = original.item_id.isin(item_ids)
        rows.append({'split': name, 'before_cohort': len(original), 'retained': len(part),
                     'outside_users': int((~known_users).sum()),
                     'outside_items': int((~known_items).sum()),
                     'outside_both': int((~known_users & ~known_items).sum()),
                     'users': part.user.nunique(), 'items': part.item.nunique(),
                     'positive_rows': int((part.rating >= 4).sum())})
    return pd.DataFrame(rows).set_index('split')
