"""Independent first-stage scheduling exercise, no author checker imports.

Read only the task, upgraded chapter and two source cards.  This script models
declared toy states; it neither implements a serving engine nor measures GPU
time. Its own small finite examples cannot establish a knowledge gain effect.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CHECKS = []


def verify(name, predicate):
    if not predicate:
        raise AssertionError(name)
    CHECKS.append(name)


@dataclass
class Request:
    name: str
    life: int
    n: int
    k: int
    pages: list[int]
    canceled: bool = False


@dataclass
class Page:
    generation: int = 0
    owner: tuple | None = None
    pins: set = field(default_factory=set)


@dataclass(frozen=True)
class Row:
    name: str
    life: int
    position: int
    sample: bool


class Pool:
    def __init__(self):
        self.pages = [Page() for _ in range(7)]
        self.requests = {'A': Request('A', 1, 5, 3, [0]),
                         'B': Request('B', 1, 8, 7, [1, 2, 3]),
                         'C': Request('C', 1, 4, 0, [])}
        for request in self.requests.values():
            for p in request.pages:
                self.pages[p].owner = (request.name, request.life)
        self.tickets = {}
        self.done = set()

    def free(self):
        return [i for i, p in enumerate(self.pages) if p.owner is None and not p.pins]

    def reserve(self, alloc):
        old = deepcopy(self.requests)
        added = []
        try:
            for name, q in alloc.items():
                request = self.requests[name]
                if not q:
                    continue
                assert not request.canceled and 0 < q <= request.n-request.k
                need = max(0, (request.k+q+2)//3-len(request.pages))
                for _ in range(need):
                    free = self.free()
                    if not free:
                        raise MemoryError('page reservation unavailable')
                    p = free[0]
                    self.pages[p].generation += 1
                    self.pages[p].owner = (name, request.life)
                    request.pages.append(p)
                    added.append(p)
            return added
        except Exception:
            for p in added:
                self.pages[p].owner = None
            self.requests = old
            raise

    def submit(self, ticket, alloc):
        assert ticket not in self.tickets and ticket not in self.done
        rows, participants, resources = [], [], []
        for name, q in alloc.items():
            if not q:
                continue
            r = self.requests[name]
            assert not r.canceled and 0 < q <= r.n-r.k
            assert (r.k+q+2)//3 <= len(r.pages)
            participants.append((name, r.life, r.k+q))
            for pos in range(r.k, r.k+q):
                rows.append(Row(name, r.life, pos, pos == r.n-1))
            for p in r.pages:
                self.pages[p].pins.add(ticket)
                resources.append((p, self.pages[p].generation))
        self.tickets[ticket] = (tuple(rows), tuple(participants), tuple(resources))
        return tuple(rows)

    def cancel(self, name):
        r = self.requests[name]
        r.canceled = True
        for p in r.pages:
            assert self.pages[p].owner == (name, r.life)
            self.pages[p].owner = None

    def complete(self, ticket):
        if ticket in self.done:
            return False
        rows, participants, resources = self.tickets[ticket]
        for name, life, end in participants:
            current = self.requests.get(name)
            if current is None or current.life != life or current.canceled:
                continue
            current.k = end
            if any(row.name == name and row.life == life and row.sample for row in rows):
                current.n += 1  # this exercise selects a non-EOS token
        for p, generation in resources:
            assert self.pages[p].generation == generation
            self.pages[p].pins.remove(ticket)
        self.done.add(ticket)
        return True


def candidate(alloc):
    initial = {'A': (5, 3, 1), 'B': (8, 7, 3), 'C': (4, 0, 0)}
    qsum, pairs, extra = 0, 0, 0
    for name, q in alloc.items():
        n, k, pages = initial[name]
        assert 0 <= q <= n-k
        qsum += q
        pairs += q*k+q*(q+1)//2
        extra += max(0, (k+q+2)//3-pages)
    return {'alloc': alloc, 'Q': qsum, 'pairs': pairs,
            'time_us': 20+2*qsum+0.5*pairs, 'incremental_pages': extra}


def main():
    rejected = candidate({'A': 2, 'B': 1, 'C': 2})
    verify('rejected candidate arithmetic',
           tuple(rejected[k] for k in ('Q', 'pairs', 'time_us', 'incremental_pages')) == (5, 20, 40, 2))
    verify('Q and pages alone would admit a time-illegal candidate',
           rejected['Q'] <= 5 and rejected['incremental_pages'] <= 3 and rejected['time_us'] > 36)
    legal = [candidate(x) for x in ({'A': 2, 'B': 1, 'C': 0},
                                    {'A': 1, 'B': 1, 'C': 1},
                                    {'A': 2, 'B': 0, 'C': 2},
                                    {'A': 0, 'B': 1, 'C': 2})]
    verify('four candidates meet all declared budgets',
           all(c['Q'] <= 5 and c['time_us'] <= 36 and c['incremental_pages'] <= 3 for c in legal))
    # An independent enumeration of causal positions checks the formula's count.
    for c in [rejected, *legal]:
        old = {'A': 3, 'B': 7, 'C': 0}
        observed = sum(1 for name, q in c['alloc'].items()
                       for position in range(old[name], old[name]+q)
                       for visible in range(position+1))
        verify('causal enumeration '+str(c['alloc']), observed == c['pairs'])
    pool = Pool()
    verify('initial ownership/free pages', len(pool.free()) == 3)
    pool.reserve(legal[0]['alloc'])
    verify('A adds one page while B needs no page',
           len(pool.requests['A'].pages) == 2 and len(pool.requests['B'].pages) == 3 and len(pool.free()) == 2)
    rows = pool.submit(('round', 0), legal[0]['alloc'])
    expected_rows = [Row('A', 1, 3, False), Row('A', 1, 4, True), Row('B', 1, 7, True)]
    verify('packed result rows retain original identity and sample boundary', list(rows) == expected_rows)
    old_b_pages = list(pool.requests['B'].pages)
    old_b_generations = [pool.pages[p].generation for p in old_b_pages]
    pool.cancel('B')
    verify('cancel does not free in-flight B pages',
           len(pool.free()) == 2 and all(pool.pages[p].pins for p in old_b_pages))
    verify('cancel leaves original snapshot unchanged', list(rows) == expected_rows)
    verify('physical completion is processed once', pool.complete(('round', 0)))
    verify('A sampled token is known but has no KV', (pool.requests['A'].n, pool.requests['A'].k) == (6, 5))
    verify('old B has no business commit after cancellation',
           (pool.requests['B'].n, pool.requests['B'].k) == (8, 7))
    verify('B pages become reusable only after completion',
           len(pool.free()) == 5 and all(not pool.pages[p].pins for p in old_b_pages))
    pool.requests['B'] = Request('B', 2, 2, 0, [])
    pool.reserve({'B': 1})
    new_b_pages = list(pool.requests['B'].pages)
    verify('new B has a distinct lifetime and physical page generation',
           new_b_pages == [old_b_pages[0]]
           and pool.pages[new_b_pages[0]].generation > old_b_generations[0])
    before = deepcopy((pool.requests, pool.pages))
    verify('late duplicate completion is rejected', not pool.complete(('round', 0)))
    verify('late completion cannot mutate new B or free its page', before == (pool.requests, pool.pages))
    early_reuse = Pool()
    early_reuse.reserve(legal[0]['alloc'])
    early_reuse.submit(('round', 0), legal[0]['alloc'])
    early_reuse.cancel('B')
    early_reuse.requests['B'] = Request('B', 2, 2, 0, [])
    early_reuse.reserve({'B': 1})
    early_new_pages = list(early_reuse.requests['B'].pages)
    verify('reused external name before old completion uses other free storage',
           not set(early_new_pages) & {1, 2, 3})
    early_reuse.complete(('round', 0))
    verify('old physical cleanup survives name reuse without committing to new B',
           early_reuse.requests['B'].n == 2 and early_reuse.requests['B'].k == 0
           and all(early_reuse.pages[p].owner == ('B', 2) for p in early_new_pages)
           and all(not early_reuse.pages[p].pins for p in (1, 2, 3)))
    # Exhaust all but one free page without changing A/C. A's first reservation
    # succeeds; C's then fails. An exception must restore the complete view.
    rollback = Pool()
    free = rollback.free()
    rollback.pages[free[1]].owner = ('other', 1)
    rollback.pages[free[2]].owner = ('other', 2)
    old_requests = deepcopy(rollback.requests)
    old_free = rollback.free()
    try:
        rollback.reserve({'A': 2, 'C': 2})
    except MemoryError:
        failed = True
    else:
        failed = False
    verify('joint A/C reservation failure is explicit', failed)
    verify('rollback restores request mapping and free capacity',
           rollback.requests == old_requests and rollback.free() == old_free and not rollback.tickets)
    verify('same Q alone predicts different device time',
           legal[0]['Q'] == legal[1]['Q'] == 3
           and legal[0]['time_us'] == 34.5 and legal[1]['time_us'] == 32.5)
    hypotheses = [
        {'cause': 'different causal histories and pairs',
         'prediction': 'same Q can have different GPU intervals even with no CPU preparation change',
         'intervention': 'prebuild equivalent views, preserve each history, time device intervals with identical Q'},
        {'cause': 'different layout/kernel/graph bucket and padding efficiency',
         'prediction': 'device-only difference persists; instructions or padded work differs',
         'intervention': 'record dispatch and valid/padded work, compare compatible fixed paths with correct outputs'},
        {'cause': 'CPU metadata or page-allocation work',
         'prediction': 'host preparation or device starvation increases, prebuilt views reduce it',
         'intervention': 'separate host intervals, submit gaps, device work and output delivery'},
        {'cause': 'contention or service waiting',
         'prediction': 'device interval or admission wait depends on co-running work/readiness',
         'intervention': 'control offered load and concurrency, retain original work amount and event boundaries'},
    ]
    paths = ['knowledge/algorithms/iteration-batching.md', 'knowledge/sources/S-ORCA.md',
             'knowledge/sources/S-SARATHI.md',
             'docs/evidence/parallel-mechanisms-20261027/tasks/scheduling-transfer.md']
    report = {'rejected': {k: v for k, v in rejected.items() if k != 'alloc'},
              'legal_candidates': legal,
              'selected_rows': [{'request': r.name, 'position': r.position, 'sample': r.sample} for r in rows],
              'A_after': {'n': 6, 'k': 5}, 'hypotheses': hypotheses,
              'documents': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              'scope': 'Independent first-stage cross-reading exercise by another author; shared parent context, no fixed budget/OS isolation, no controlled knowledge-gain conclusion; no upstream code or author checks/reports read.',
              'date': '2026-10-28', 'status': 'PASS', 'checks_passed': len(CHECKS), 'checks': CHECKS,
              'solution_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['Synthetic time model; no GPU timing.',
                         'Toy sequential coordinator does not prove concurrent production engine correctness.',
                         'No claim of causal Agent benefit or general transfer success.']}
    (HERE/'scheduling-answer.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checks_passed': len(CHECKS),
                      'rejected': report['rejected'], 'A_after': report['A_after'],
                      'chapter_sha256': report['documents'][paths[0]]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
