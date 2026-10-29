import json
import math
from collections import namedtuple

# Standalone binary64 CPU model; no file, process, project or network access.
Row = namedtuple('Row', 'logical coord key value identity')
IDENTITY = 'fixed-toy-model-and-position-contract'
D = 4
DV = 3
FREQ = tuple(10000.0 ** (-i / D) for i in range(0, D, 2))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def qkv(token):
    # Fixed externally specified raw inputs, not a Transformer implementation.
    q = tuple((((token + 1) * (i + 2)) % 11 - 5) / 4.0 for i in range(D))
    k = tuple((((token + 3) * (i + 1)) % 13 - 6) / 5.0 for i in range(D))
    v = (float(token), float((token + 1) % 4) - 1.5, math.sin(token))
    return q, k, v


def rotate(x, position):
    result = []
    for pair, frequency in enumerate(FREQ):
        i = 2 * pair
        angle = position * frequency
        c, s = math.cos(angle), math.sin(angle)
        result.extend((c * x[i] - s * x[i + 1], s * x[i] + c * x[i + 1]))
    return tuple(result)


def reference(tokens, coords, window):
    # Independent visibility scan and relative-angle score expansion.
    # Does not call rotate(), Store, page lookup, or the DUT mask routine.
    result = []
    for t, token in enumerate(tokens):
        q, _, _ = qkv(token)
        scores, values = [], []
        for j, key_token in enumerate(tokens):
            if j > t:
                continue
            if window is not None and t - j >= window:
                continue
            _, k, v = qkv(key_token)
            score = 0.0
            for pair, frequency in enumerate(FREQ):
                i = 2 * pair
                angle = (coords[j] - coords[t]) * frequency
                score += math.cos(angle) * (q[i] * k[i] + q[i + 1] * k[i + 1])
                score += math.sin(angle) * (q[i + 1] * k[i] - q[i] * k[i + 1])
            scores.append(score / math.sqrt(D))
            values.append(v)
        require(bool(scores), 'reference valid row lacks its own key')
        maximum = max(scores)
        weights = [math.exp(z - maximum) for z in scores]
        denominator = math.fsum(weights)
        result.append(tuple(math.fsum(w * v[d] for w, v in zip(weights, values)) / denominator
                            for d in range(DV)))
    return result


def close_rows(actual, expected):
    require(len(actual) == len(expected), 'missing or extra query rows')
    for i, (a, e) in enumerate(zip(actual, expected)):
        require(len(a) == len(e), 'output width')
        for d, (x, y) in enumerate(zip(a, e)):
            require(math.isfinite(x), 'non-finite output')
            require(abs(x - y) <= 1e-10 * (1.0 + abs(y)),
                    'output mismatch at row %d channel %d: %.17g vs %.17g' % (i, d, x, y))


class Pool:
    def __init__(self, page_size):
        self.page_size = page_size
        self.pages = {}
        self.frozen = set()

    def allocate(self):
        physical = 19 + 11 * len(self.pages)
        self.pages[physical] = [None] * self.page_size
        return physical


class Store:
    def __init__(self, page_size, window, representation, pool=None):
        self.page_size = page_size
        self.window = window
        self.representation = representation
        self.pool = pool if pool is not None else Pool(page_size)
        self.table = {}
        self.ring = {}
        self.end = 0
        self.copied_valid_slots = 0

    def put(self, row):
        t = row.logical
        if self.window is not None:
            self.ring[t % self.window] = row
            return
        logical_page, offset = divmod(t, self.page_size)
        if logical_page not in self.table:
            self.table[logical_page] = self.pool.allocate()
        physical = self.table[logical_page]
        if physical in self.pool.frozen:
            replacement = self.pool.allocate()
            # Copy from the old physical page, not a full logical repack.
            old_cells = self.pool.pages[physical]
            for slot, old_row in enumerate(old_cells):
                if old_row is not None:
                    self.pool.pages[replacement][slot] = old_row
                    self.copied_valid_slots += 1
            self.table[logical_page] = replacement
            physical = replacement
        self.pool.pages[physical][offset] = row

    def get(self, t, expected_coord, expected_identity):
        if self.window is not None:
            row = self.ring.get(t % self.window)
        else:
            logical_page, offset = divmod(t, self.page_size)
            physical = self.table.get(logical_page)
            row = None if physical is None else self.pool.pages[physical][offset]
        if row is None:
            raise ValueError('missing logical key')
        if row.logical != t:
            raise ValueError('stale ring or page tag')
        if row.coord != expected_coord:
            raise ValueError('position metadata mismatch')
        if row.identity != expected_identity:
            raise ValueError('content identity mismatch')
        return row

    def fork(self):
        require(self.window is None, 'this COW test covers full paged storage only')
        self.pool.frozen.update(self.table.values())
        child = Store(self.page_size, None, self.representation, self.pool)
        child.table = dict(self.table)
        child.end = self.end
        return child

    def relocate(self):
        require(self.window is None, 'relocation test uses pages')
        old_mapping = sorted(self.table.items())
        new_physical = [self.pool.allocate() for _ in old_mapping]
        for (logical_page, source), target in zip(old_mapping, reversed(new_physical)):
            self.pool.pages[target][:] = self.pool.pages[source][:]
            self.table[logical_page] = target


def evaluate(q, coord, columns, t, local_i, window, representation, fault=None):
    q_position = local_i if fault == 'block_reset' else coord
    rotated_q = rotate(q, q_position)
    scores, values = [], []
    entries = list(columns)
    if fault == 'duplicate_self':
        own = next(row for row, valid in columns if valid and row.logical == t)
        entries.append((own, True))
    for column_i, (row, valid) in enumerate(entries):
        allowed = valid and row.logical <= t
        if window is not None:
            allowed = allowed and row.logical >= t - window + 1
        if fault == 'upper_left':
            allowed = valid and column_i <= local_i
        elif fault == 'ignore_valid':
            allowed = row.logical <= t
            if window is not None:
                allowed = allowed and row.logical >= t - window + 1
        elif fault == 'future':
            allowed = valid
            if window is not None:
                allowed = allowed and row.logical >= t - window + 1
        if not allowed and fault != 'multiply_zero':
            continue
        key = row.key if representation == 'rotated' else rotate(row.key, row.coord)
        if fault == 'double_rotate':
            key = rotate(key, row.coord)
        score = sum(x * y for x, y in zip(rotated_q, key)) / math.sqrt(D)
        if not allowed:
            score *= 0.0
        scores.append(score)
        values.append(row.value)
    # Explicit low-level adapter convention; valid causal rows must be nonempty.
    if not scores:
        return (0.0,) * DV
    maximum = max(scores)
    weights = [math.exp(z - maximum) for z in scores]
    denominator = sum(weights)
    return tuple(sum(w * v[d] for w, v in zip(weights, values)) / denominator
                 for d in range(DV))


def advance(store, tokens, coords, start, end, padding=0, fault=None):
    require(start == store.end, 'wrong committed endpoint')
    require(0 <= start <= end <= len(tokens), 'invalid token range')
    require(len(coords) == len(tokens), 'coordinate range')
    if start == end:
        return []
    staged = []
    for t in range(start, end):
        _, key, value = qkv(tokens[t])
        coordinate = t - start if fault == 'block_reset' else coords[t]
        rotation_position = coordinate
        if fault == 'slot_coord':
            rotation_position = t % (store.window or store.page_size)
        if store.representation == 'rotated':
            key = rotate(key, rotation_position)
        staged.append(Row(t, coordinate, key, value, IDENTITY))
    if fault == 'early_ring':
        require(store.window is not None, 'early overwrite needs ring storage')
        for row in staged:
            store.put(row)
    lower = 0 if store.window is None else max(0, start - store.window + 1)
    old = [store.get(t, coords[t], IDENTITY) for t in range(lower, start)]
    poison = Row(0, 0, (0.0,) * D, (1000000.0, -2000000.0, 3000000.0), 'unused')
    # Both left padding and unused static capacity have explicit invalidity.
    columns = [(poison, False)] * padding
    columns += [(row, True) for row in old + staged]
    columns += [(poison, False), (poison, False)]
    output = []
    for t in range(start, end):
        q, _, _ = qkv(tokens[t])
        output.append(evaluate(q, coords[t], columns, t, t - start,
                               store.window, store.representation, fault))
    # All chunk consumers finish before ring replacement or normal commit.
    for row in staged:
        store.put(row)
    store.end = end
    return output


def compositions(n):
    if n == 0:
        yield ()
        return
    for first in range(1, n + 1):
        for rest in compositions(n - first):
            yield (first,) + rest


def schemes(n):
    if n <= 5:
        return list(compositions(n))
    result = {(n,), (1,) * n}
    if n == 7:
        result.update(((2, 2, 3), (5, 2)))
    if n == 10:
        result.update(((3, 4, 3), (6, 4)))
    return sorted(result)


def expect_rejected(name, operation):
    try:
        operation()
    except (AssertionError, ValueError):
        return name
    raise AssertionError('negative control was not detected: ' + name)


def main():
    paths = 0
    query_rows = 0
    for n in (0, 1, 2, 5, 7, 10):
        tokens = [2 * i + 1 for i in range(n)]
        coordinate_sets = (list(range(n)), [9 + i for i in range(n)],
                           [9 + (5 * i) % 17 for i in range(n)])
        for coords in coordinate_sets:
            for window in (None, 1, 3, 4):
                expected = reference(tokens, coords, window)
                for page_size in (2, 4):
                    for representation in ('raw', 'rotated'):
                        for padding in (0, 3):
                            for chunks in schemes(n):
                                store = Store(page_size, window, representation)
                                observed = []
                                cursor = 0
                                for chunk in chunks:
                                    observed.extend(advance(store, tokens, coords, cursor,
                                                            cursor + chunk, padding))
                                    cursor += chunk
                                require(cursor == n and store.end == n, 'partition endpoint')
                                close_rows(observed, expected)
                                old_table = dict(store.table)
                                old_ring = dict(store.ring)
                                old_page_count = len(store.pool.pages)
                                require(advance(store, tokens, coords, n, n, padding) == [],
                                        'empty append result')
                                require(store.table == old_table and store.ring == old_ring
                                        and len(store.pool.pages) == old_page_count
                                        and store.end == n, 'empty append changed state')
                                paths += 1
                                query_rows += n

    # Actual shared partial-page forks, followed by consuming each private suffix.
    cow_cases = 0
    for representation in ('raw', 'rotated'):
        base_tokens = [1, 3, 5, 7, 9]
        base_coords = [12 + i for i in range(5)]
        base = Store(4, None, representation)
        close_rows(advance(base, base_tokens, base_coords, 0, 5, 3),
                   reference(base_tokens, base_coords, None))
        source_pages = {pid: tuple(base.pool.pages[pid]) for pid in base.table.values()}
        a, b = base.fork(), base.fork()
        tokens_a = base_tokens + [61, 62, 63, 64]
        tokens_b = base_tokens + [81, 82]
        coords_a = [12 + i for i in range(len(tokens_a))]
        coords_b = [12 + i for i in range(len(tokens_b))]
        close_rows(advance(a, tokens_a, coords_a, 5, 8, 3),
                   reference(tokens_a, coords_a, None)[5:8])
        close_rows(advance(b, tokens_b, coords_b, 5, 7, 0),
                   reference(tokens_b, coords_b, None)[5:7])
        require(a.table[0] == b.table[0] == base.table[0], 'full prefix page lost sharing')
        require(len({a.table[1], b.table[1], base.table[1]}) == 3, 'tail page write alias')
        require(a.copied_valid_slots == b.copied_valid_slots == 1,
                'COW must copy the one old valid physical slot')
        for pid, cells in source_pages.items():
            require(tuple(base.pool.pages[pid]) == cells, 'shared source page changed')
        for t in range(5):
            base.get(t, base_coords[t], IDENTITY)
        # Relabel and reverse physical placement, then consume the moved history.
        prior_table = dict(a.table)
        a.relocate()
        require(a.table != prior_table, 'relocation did no work')
        close_rows(advance(a, tokens_a, coords_a, 8, 9, 3),
                   reference(tokens_a, coords_a, None)[8:9])
        for pid, cells in source_pages.items():
            require(tuple(base.pool.pages[pid]) == cells, 'relocation changed source')
        cow_cases += 1

    # Explicit non-square adapter contracts, including all-masked rows.
    alignment_cases = (
        (2, 5, [[0, 1, 2, 3], [0, 1, 2, 3, 4]]),
        (5, 2, [[], [], [], [0], [0, 1]]),
        (1, 7, [[0, 1, 2, 3, 4, 5, 6]]),
    )
    for length_q, length_k, expected in alignment_cases:
        observed = [[j for j in range(length_k) if j <= i + length_k - length_q]
                    for i in range(length_q)]
        require(observed == expected, 'right-bottom adapter')
    require(evaluate((1.0,) * D, 9, [], 0, 0, None, 'raw') == (0.0,) * DV,
            'explicit empty-row adapter')
    tokens = [1, 4, 7, 10, 13]
    close_rows(reference(tokens, list(range(5)), None),
               reference(tokens, [31 + i for i in range(5)], None))

    # Equal-score window set oracle, independent of ring addressing.
    sets = [[j for j in range(8) if j <= t and t - j < 4] for t in (5, 6, 7)]
    require(sets == [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]], 'window endpoints')
    means = [sum(row) / len(row) for row in sets]
    require(means == [3.5, 4.5, 5.5], 'window contribution reference')

    detected = []
    faults = ('upper_left', 'multiply_zero', 'ignore_valid', 'double_rotate',
              'block_reset', 'slot_coord', 'future', 'duplicate_self', 'early_ring')
    for fault in faults:
        def wrong_path(fault=fault):
            window = 4 if fault == 'early_ring' else None
            tokens = [1, 3, 5, 7, 9, 61, 62, 63]
            coords = [9 + i for i in range(len(tokens))]
            store = Store(4, window, 'rotated')
            advance(store, tokens, coords, 0, 5, 3)
            observed = advance(store, tokens, coords, 5, 8, 3, fault)
            close_rows(observed, reference(tokens, coords, window)[5:8])
        detected.append(expect_rejected(fault, wrong_path))

    for field in ('logical', 'coord', 'identity'):
        def broken_metadata(field=field):
            store = Store(4, None, 'rotated')
            advance(store, [1, 3], [9, 10], 0, 2)
            physical = store.table[0]
            row = store.pool.pages[physical][0]
            replacement = 'wrong-model' if field == 'identity' else getattr(row, field) + 100
            store.pool.pages[physical][0] = row._replace(**{field: replacement})
            store.get(0, 9, IDENTITY)
        detected.append(expect_rejected('metadata_' + field, broken_metadata))

    pairs = sum(t + 1 for t in range(5, 7))
    require(pairs == 2 * 5 + 2 * 3 // 2 == 13, 'pair count formula')
    token_bytes = 32 * 8 * (128 + 128) * 2
    require(token_bytes == 131072 and 4 * token_bytes == 524288, 'KV capacity arithmetic')
    require((2 * 1 + 2) * token_bytes == 524288, 'COW traffic arithmetic')
    print(json.dumps({
        'scope': 'deterministic stdlib binary64 single-layer CPU model',
        'partition_storage_paths': paths,
        'query_rows_compared_to_independent_relative_angle_reference': query_rows,
        'shared_partial_page_forks_and_following_relocation': cow_cases,
        'negative_controls_detected': detected,
        'equal_score_window_outputs': means,
        'valid_pairs_H5_q2': pairs,
        'KV_bytes_per_token_all_32_layers_in_cost_example': token_bytes,
        'not_covered': ['GPU measurements or memory visibility', 'full Transformer',
                        'multilayer atomic commit', 'asynchronous reclamation',
                        'quantization accuracy or task quality', 'performance']
    }, sort_keys=True))


if __name__ == '__main__':
    main()
