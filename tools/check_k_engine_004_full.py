from itertools import product


class Node:
    def __init__(self, label=(), pages=()):
        self.label = tuple(label)
        self.pages = tuple(pages)
        self.parent = None
        self.children = {}
        self.holders = set()
        self.last = 0
        self.live = True


def common_len(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def attach(parent, child):
    assert child.label, 'empty edge'
    assert child.label[0] not in parent.children, 'duplicate child key'
    parent.children[child.label[0]] = child
    child.parent = parent


def detach(child):
    parent = child.parent
    assert parent is not None, 'detach root'
    key = child.label[0]
    assert parent.children.get(key) is child, 'child index mismatch'
    del parent.children[key]
    child.parent = None


class Tree:
    def __init__(self, mutant=None):
        self.roots = {}
        self.tickets = {}
        self.clock = 0
        self.mutant = mutant

    def root(self, ns):
        if ns not in self.roots:
            self.roots[ns] = Node()
        return self.roots[ns]

    def nodes(self):
        def walk(ns, node, prefix):
            full = prefix + node.label
            yield ns, node, full
            for child in node.children.values():
                yield from walk(ns, child, full)
        for ns, root in self.roots.items():
            yield from walk(ns, root, ())

    def split(self, child, k):
        assert 0 < k < len(child.label), 'invalid split'
        assert not child.holders, 'active edge split'
        parent = child.parent
        label, pages = child.label, child.pages
        middle = Node(label[:k], pages[:k])
        middle.last = child.last
        if self.mutant == 'rename_before_detach':
            child.label, child.pages = label[k:], pages[k:]
            detach(child)
        else:
            detach(child)
            child.label, child.pages = label[k:], pages[k:]
        attach(parent, middle)
        attach(middle, child)
        return middle

    def lookup(self, ns, seq):
        seq = tuple(seq)
        node, pos = self.root(ns), 0
        while pos < len(seq):
            child = node.children.get(seq[pos])
            if child is None:
                break
            k = common_len(child.label, seq[pos:])
            pos += k
            if k < len(child.label):
                break
            node = child
        return pos

    def insert(self, ns, seq, pages, ready=True):
        seq, pages = tuple(seq), tuple(pages)
        assert len(seq) == len(pages), 'page width mismatch'
        if not ready:
            return False
        node, pos = self.root(ns), 0
        while pos < len(seq):
            child = node.children.get(seq[pos])
            if child is None:
                self._append(node, seq[pos:], pages[pos:])
                return True
            k = common_len(child.label, seq[pos:])
            assert k > 0
            if k < len(child.label):
                if child.holders:
                    return False
                node = self.split(child, k)
                pos += k
                if pos < len(seq):
                    self._append(node, seq[pos:], pages[pos:])
                return True
            node, pos = child, pos + k
        return True

    def _append(self, parent, label, pages):
        occupied = {p for _, n, _ in self.nodes() for p in n.pages}
        assert len(set(pages)) == len(pages), 'duplicate new physical page'
        assert not occupied.intersection(pages), 'new page already cached'
        attach(parent, Node(label, pages))

    def _boundary(self, ns, seq, endpoint):
        node, pos = self.root(ns), 0
        path = [node]
        while pos < endpoint:
            child = node.children[seq[pos]]
            if endpoint - pos < len(child.label):
                child = self.split(child, endpoint - pos)
            assert seq[pos:pos + len(child.label)] == child.label
            node = child
            pos += len(child.label)
            path.append(node)
        assert pos == endpoint
        return tuple(path)

    def acquire(self, ns, seq, tid, allowed=None, upper=None):
        assert tid not in self.tickets, 'ticket reused'
        seq = tuple(seq)
        raw = self.lookup(ns, seq)
        upper = len(seq) if upper is None else upper
        allowed = set(range(raw + 1)) if allowed is None else set(allowed)
        choices = {0}
        node, pos = self.root(ns), 0
        while pos < raw:
            child = node.children[seq[pos]]
            k = min(len(child.label), raw - pos)
            if not child.holders:
                choices.update(range(pos + 1, pos + k + 1))
            if k == len(child.label):
                choices.add(pos + k)
            else:
                break
            node, pos = child, pos + k
        choices = {p for p in choices if p in allowed and p <= upper}
        if not choices:
            return None, None
        endpoint = max(choices)
        path = self._boundary(ns, seq, endpoint)
        self.clock += 1
        protected = path[-1:] if self.mutant == 'omit_ancestors' else path
        for node in protected:
            node.holders.add(tid)
        for node in path:
            node.last = self.clock
        self.tickets[tid] = {
            'ns': ns, 'seq': seq, 'path': path, 'active': True
        }
        return endpoint, path

    def release(self, tid):
        ticket = self.tickets[tid]
        if not ticket['active']:
            return
        path = ticket['path']
        if self.mutant == 'release_rematched_path':
            endpoint = self.lookup(ticket['ns'], ticket['seq'])
            path = self._boundary(ticket['ns'], ticket['seq'], endpoint)
        for node in path:
            assert tid in node.holders, 'release of an unacquired node'
            node.holders.remove(tid)
        ticket['active'] = False

    def evict(self, need, unsafe=False):
        freed = []
        while len(freed) < need:
            candidates = [
                (node.last, ns, full, node)
                for ns, node, full in self.nodes()
                if node.parent is not None and not node.children
                and (unsafe or not node.holders)
            ]
            if not candidates:
                break
            _, _, _, node = min(candidates, key=lambda item: item[:3])
            detach(node)
            node.live = False
            freed.extend(node.pages)
        return tuple(freed)

    def reclaimable_pages(self):
        return {
            p for _, node, _ in self.nodes()
            if not node.holders for p in node.pages
        }


def audit(tree, holds, expected=None):
    actual = {}
    physical = []
    for ns, node, full in tree.nodes():
        assert node.live, 'retired node is discoverable'
        assert len(node.label) == len(node.pages), 'label/page mismatch'
        if node.parent is None:
            assert not node.label and not node.pages, 'nonempty root'
        else:
            assert node.label, 'empty nonroot edge'
            assert node.parent.children.get(node.label[0]) is node
        for key, child in node.children.items():
            assert child.parent is node and child.label[0] == key, 'index mismatch'
        wanted = {
            tid for tid, (owner_ns, target) in holds.items()
            if owner_ns == ns and len(full) <= len(target)
            and target[:len(full)] == full
        }
        assert node.holders == wanted, 'logical ancestor protection mismatch'
        parent_length = len(full) - len(node.label)
        for j, page in enumerate(node.pages, 1):
            key = (ns, full[:parent_length + j])
            assert key not in actual, 'duplicate logical prefix'
            actual[key] = page
            physical.append(page)
    assert len(physical) == len(set(physical)), 'cache page counted twice'
    for ns, prefix in actual:
        for k in range(1, len(prefix)):
            assert (ns, prefix[:k]) in actual, 'missing cached ancestor'
    for ns, target in holds.values():
        for k in range(1, len(target) + 1):
            assert (ns, target[:k]) in actual, 'protected state disappeared'
    if expected is not None:
        assert actual == expected, 'independent prefix/page reference mismatch'
    return actual


def add_reference(reference, ns, seq, pages):
    seq = tuple(seq)
    for k, page in enumerate(pages, 1):
        reference.setdefault((ns, seq[:k]), page)


def oracle_lcp(reference, ns, seq):
    seq = tuple(seq)
    return max(
        (k for k in range(len(seq) + 1)
         if k == 0 or (ns, seq[:k]) in reference),
        default=0
    )


def admit(tree, ns, seq, tid, free, private_count, allowed=None, upper=None):
    assert private_count >= 0
    endpoint, _ = tree.acquire(ns, seq, tid, allowed, upper)
    if endpoint is None:
        return None, False, (), ()
    cold = tree.reclaimable_pages()
    if tree.mutant == 'protected_capacity':
        cold = {p for _, node, _ in tree.nodes() for p in node.pages}
    if private_count > len(free) + len(cold):
        tree.release(tid)
        return endpoint, False, (), ()
    freed = tree.evict(max(0, private_count - len(free)))
    free.update(freed)
    assert len(free) >= private_count, 'physical reservation impossible'
    allocated = tuple(sorted(free)[:private_count])
    free.difference_update(allocated)
    return endpoint, True, allocated, freed


def audit_pool(tree, holds, total, free, private):
    cached = set(audit(tree, holds).values())
    groups = [cached, set(free)] + [set(pages) for pages in private.values()]
    union = set().union(*groups)
    assert sum(map(len, groups)) == len(union), 'physical roles overlap'
    assert union == set(total), 'pool conservation failed'


def test_prefix_reference():
    words = [tuple(s) for s in ('abcd', 'abx', 'a', 'ab', 'x', 'abce')]
    queries = [
        word for width in range(4)
        for word in product('abcx', repeat=width)
    ] + words
    comparisons = 0
    for rotation in range(len(words)):
        tree, reference = Tree(), {}
        page = rotation * 1000
        order = words[rotation:] + words[:rotation]
        for step, seq in enumerate(order):
            pages = tuple(range(page, page + len(seq)))
            page += len(seq)
            assert tree.insert('n', seq, pages)
            add_reference(reference, 'n', seq, pages)
            audit(tree, {}, reference)
            for query in queries:
                assert tree.lookup('n', query) == oracle_lcp(reference, 'n', query)
                comparisons += 1
            query = seq[:-1]
            tid = ('probe', rotation, step)
            endpoint, _ = tree.acquire('n', query, tid)
            assert endpoint == oracle_lcp(reference, 'n', query)
            audit(tree, {tid: ('n', query[:endpoint])}, reference)
            tree.release(tid)
            tree.release(tid)
            audit(tree, {}, reference)
        assert tree.lookup('other', words[0]) == 0
        audit(tree, {}, reference)
    print('prefix_reference_comparisons', comparisons)


def test_concrete_split_and_active_edge():
    tree, reference = Tree(), {}
    seq, pages = tuple('abcd'), (10, 11, 12, 13)
    assert tree.insert('n', seq, pages)
    add_reference(reference, 'n', seq, pages)
    endpoint, _ = tree.acquire('n', tuple('abx'), 'split')
    assert endpoint == 2
    node = tree.root('n').children['a']
    assert node.label == tuple('ab') and node.pages == (10, 11)
    assert node.children['c'].pages == (12, 13)
    audit(tree, {'split': ('n', tuple('ab'))}, reference)
    tree.release('split')
    audit(tree, {}, reference)

    tree = Tree()
    assert tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
    endpoint, _ = tree.acquire('n', tuple('abcd'), 'A')
    assert endpoint == 4
    holds = {'A': ('n', tuple('abcd'))}
    before = audit(tree, holds)
    endpoint, _ = tree.acquire('n', tuple('abx'), 'B')
    assert endpoint == 0
    holds['B'] = ('n', ())
    assert not tree.insert('n', tuple('abx'), (20, 21, 22))
    assert not tree.insert('n', tuple('a'), (23,))
    audit(tree, holds, before)
    tree.release('A')
    del holds['A']
    endpoint, _ = tree.acquire('n', tuple('abx'), 'C')
    assert endpoint == 2
    holds['C'] = ('n', tuple('ab'))
    audit(tree, holds, before)
    tree.release('B')
    tree.release('C')
    assert tree.insert('n', tuple('abx'), (30, 31, 32))
    after = dict(before)
    after[('n', tuple('abx'))] = 32
    audit(tree, {}, after)
    unchanged = audit(tree, {})
    assert not tree.insert('n', tuple('zz'), (50, 51), ready=False)
    audit(tree, {}, unchanged)
    print('split_and_active_edge', 'page mapping preserved; active insertion deferred')


def test_qualification():
    tree = Tree()
    seq = tuple(range(12))
    assert tree.insert('n', seq, tuple(range(100, 112)))
    groups = [{0, 4, 8, 12}, {0, 8, 12}, {0, 8}]
    allowed = set.intersection(*groups)
    independent = max(p for p in range(12) if all(p in g for g in groups))
    endpoint, _ = tree.acquire('n', seq, 'joint', allowed, 11)
    assert endpoint == independent == 8
    audit(tree, {'joint': ('n', seq[:8])})
    tree.release('joint')
    endpoint, _ = tree.acquire('n', seq[:8], 'last_row', {0, 4, 8}, 7)
    assert endpoint == 4
    tree.release('last_row')
    groups = [{0, 4, 8, 12}, {0, 6, 10}]
    endpoint, _ = tree.acquire('n', seq, 'holes', set.intersection(*groups), 12)
    assert endpoint == 0
    assert min(map(max, groups)) == 10
    tree.release('holes')
    endpoint, _ = tree.acquire('n', seq[:1], 'single', {0, 1}, 0)
    assert endpoint == 0
    tree.release('single')
    endpoint, _ = tree.acquire('n', (), 'empty', {0}, 0)
    assert endpoint == 0
    tree.release('empty')
    assert tree.acquire('n', seq, 'no_checkpoint', set(), 12) == (None, None)
    audit(tree, {})
    print('qualified_positions', 8, 4, 0, 'empty intersection rejected')


def test_capacity_and_publication():
    tree = Tree()
    assert tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
    free, private, holds = {4, 5}, {}, {}
    p, ok, allocated, freed = admit(tree, 'n', tuple('abx'), 'B', free, 1, upper=2)
    assert (p, ok, allocated, freed) == (2, True, (4,), ())
    holds['B'] = ('n', tuple('ab'))
    private['B'] = set(allocated)
    audit_pool(tree, holds, range(6), free, private)
    p, ok, allocated, freed = admit(tree, 'n', tuple('uvw'), 'D', free, 3)
    assert (p, ok, allocated) == (0, True, (2, 3, 5))
    assert set(freed) == {2, 3}
    holds['D'] = ('n', ())
    private['D'] = set(allocated)
    audit_pool(tree, holds, range(6), free, private)
    assert tree.lookup('n', tuple('abcd')) == 2
    assert tree.evict(20) == ()
    audit_pool(tree, holds, range(6), free, private)

    # Normal completion with all physical accesses already ended.
    assert tree.insert('n', tuple('abx'), (0, 1, 4))
    private['B'].remove(4)
    tree.release('B')
    del holds['B']
    tree.release('B')
    audit_pool(tree, holds, range(6), free, private)
    tree.release('D')
    del holds['D']
    audit_pool(tree, holds, range(6), free, private)

    # Recursive leaves include ancestors that become leaves later.
    tree = Tree()
    assert tree.insert('n', tuple('ab'), (10, 11))
    assert tree.insert('n', tuple('abcd'), (10, 11, 12, 13))
    assert tree.reclaimable_pages() == {10, 11, 12, 13}
    assert set(tree.evict(4)) == {10, 11, 12, 13}
    assert tree.evict(0) == ()
    audit(tree, {}, {})

    tree = Tree()
    assert tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
    free = set()
    p, ok, allocated, freed = admit(tree, 'n', tuple('abcde'), 'reject', free, 1)
    assert (p, ok, allocated, freed) == (4, False, (), ())
    assert free == set()
    audit(tree, {})
    print('capacity_observation', 'D reserves pages 2,3,5; protected pages remain resident')


def test_schedule_boundary():
    def new_tokens(order):
        previous, work = '', 0
        for seq in order:
            work += len(seq) - common_len(seq, previous)
            previous = seq
        return work
    words = ('abc', 'xyz', 'abq')
    prefixes = {word[:k] for word in words for k in range(1, len(word) + 1)}
    static = new_tokens(words)
    updated = new_tokens(('abc', 'abq', 'xyz'))
    assert static == 9 and updated == len(prefixes) == 7
    print('single_path_capacity_three', 'static_once', static, 'updated_order', updated)


def detect(name, operation):
    try:
        operation()
    except AssertionError as error:
        print('negative_control_detected', name, str(error))
    else:
        raise AssertionError('negative control escaped: ' + name)


def test_negative_controls():
    def old_split_order():
        tree = Tree('rename_before_detach')
        tree.insert('n', tuple('abcd'), (10, 11, 12, 13))
        tree.acquire('n', tuple('abx'), 'B')
        audit(tree, {'B': ('n', tuple('ab'))})

    def missing_ancestor():
        tree = Tree('omit_ancestors')
        tree.insert('n', tuple('ab'), (0, 1))
        tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
        tree.acquire('n', tuple('abcd'), 'A')
        audit(tree, {'A': ('n', tuple('abcd'))})

    def protected_eviction():
        tree = Tree()
        tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
        tree.acquire('n', tuple('abcd'), 'A')
        tree.evict(4, unsafe=True)
        audit(tree, {'A': ('n', tuple('abcd'))})

    def protected_capacity():
        tree = Tree('protected_capacity')
        tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
        admit(tree, 'n', tuple('abcde'), 'A', set(), 1)

    def release_new_path():
        tree = Tree('release_rematched_path')
        tree.insert('n', tuple('abcd'), (0, 1, 2, 3))
        endpoint, _ = tree.acquire('n', tuple('abx'), 'B')
        assert endpoint == 2
        tree.insert('n', tuple('abx'), (0, 1, 4))
        tree.release('B')
        audit(tree, {})

    detect('recorded_rename_before_detach', old_split_order)
    detect('missing_ancestor_protection', missing_ancestor)
    detect('protected_state_eviction', protected_eviction)
    detect('protected_pages_as_available', protected_capacity)
    detect('release_after_rematching_publication', release_new_path)


def main():
    test_prefix_reference()
    test_concrete_split_and_active_edge()
    test_qualification()
    test_capacity_and_publication()
    test_schedule_boundary()
    test_negative_controls()
    print('scope', 'deterministic CPU symbolic tree and page sets only')
    print('limits', 'no numerical KV, logits, GPU, DMA, concurrency, upstream execution or timing')
    print('checks_completed_without_assertion')


if __name__ == '__main__':
    main()
