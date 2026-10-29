# K-SYS-014: finite CPU protocol and recovery arithmetic checks.
# Standalone, deterministic, offline, read-only; reads only this script for hashing.
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import hashlib
import json
import sys

DATA = {
    'actions': ['old', 'freeze', 'grant', 'new', 'cancel', 'fail'],
    'tokens': [2, 1, 3, 0],
    'required_ranks': [0, 1],
    'group_incarnation': 4,
    'growth': {'D_MiB': 1000, 'B_MiB_s': 100, 'g_MiB_s': 10, 'online_stages': 3},
    'kv': {'layers': 32, 'heads': 8, 'width': 128, 'element_bytes': 2, 'tokens': 4096},
}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


class Gate:
    def __init__(self, mutant=None):
        self.mutant = mutant
        self.owner = 'src'
        self.e = 0
        self.v = 0
        self.phase = 'LIVE'
        self.terminal = None
        self.group_ok = True
        self.out = []
        self.commits = {}
        self.decisions = {}
        self.frozen_tx = None
        self.snapshot = None

    def view(self):
        return (self.owner, self.e, self.v, self.phase, self.terminal,
                self.group_ok, tuple(self.out), tuple(sorted(self.commits.items())))

    def append(self, cid, owner, epoch, parent, token, ranks=(0, 1), group=4):
        fp = (owner, epoch, parent, token, tuple(sorted(ranks)), group)
        if cid in self.commits:
            if self.commits[cid] != fp:
                return False
            if self.mutant == 'duplicate_install':
                self.out.append(token)
                self.v += 1
            return True
        if self.terminal is not None or not self.group_ok or self.phase != 'LIVE':
            return False
        if self.mutant != 'parent' and parent != self.v:
            return False
        if self.mutant != 'fence' and (owner != self.owner or epoch != self.e):
            return False
        if self.mutant != 'ranks':
            if tuple(sorted(ranks)) != tuple(DATA['required_ranks']):
                return False
            if group != DATA['group_incarnation']:
                return False
        self.out.append(token)
        self.v += 1
        self.commits[cid] = fp
        return True

    def freeze(self, tx='t'):
        key = ('freeze', tx)
        if key in self.decisions:
            return True
        if self.terminal is not None or not self.group_ok:
            return False
        if self.phase != 'LIVE' or self.owner != 'src':
            return False
        self.owner = None
        self.e += 1
        self.phase = 'FROZEN'
        self.frozen_tx = tx
        self.snapshot = (self.v, tuple(self.out))
        self.decisions[key] = self.snapshot
        return True

    def grant(self, tx='t', snapshot_version=None, families=None):
        if self.snapshot is None:
            return False
        if snapshot_version is None:
            snapshot_version = self.snapshot[0]
        p = len(self.snapshot[1])
        if families is None:
            families = {('KV', p), ('REC', p)}
        fp = (snapshot_version, tuple(sorted(families)))
        key = ('grant', tx)
        if key in self.decisions:
            return self.decisions[key] == fp
        if self.terminal is not None or not self.group_ok:
            return False
        if self.phase != 'FROZEN' or self.owner is not None or tx != self.frozen_tx:
            return False
        if snapshot_version != self.v or snapshot_version != self.snapshot[0]:
            return False
        if self.mutant != 'ready' and families != {('KV', p), ('REC', p)}:
            return False
        self.owner = 'dst'
        self.e += 1
        self.phase = 'LIVE'
        self.decisions[key] = fp
        return True

    def cancel(self):
        if self.terminal is None:
            self.terminal = 'CANCELLED'
            self.owner = None
            self.e += 1
            self.phase = 'TERMINAL'

    def fail(self):
        if self.group_ok:
            self.group_ok = False
            self.owner = None
            self.e += 1
            if self.terminal is None:
                self.phase = 'RECOVERING'


def check_orderings():
    count = 0
    for trace in permutations(DATA['actions']):
        m = Gate()
        for event in trace:
            if event == 'old':
                m.append('a', 'src', 0, 0, 7)
            elif event == 'freeze':
                m.freeze()
            elif event == 'grant':
                m.grant()
            elif event == 'new':
                m.append('b', 'dst', 2, 0, 9)
            elif event == 'cancel':
                m.cancel()
            else:
                m.fail()
        # Independent oracle: inequalities over the original event positions.
        # It does not call Gate or replay its state-transition implementation.
        at = {event: i for i, event in enumerate(trace)}
        stop = min(at['cancel'], at['fail'])
        old_ok = at['old'] < min(at['freeze'], stop)
        new_ok = (at['freeze'] < at['grant'] < at['new'] < stop and not old_ok)
        expected = [7] if old_ok else ([9] if new_ok else [])
        require(m.out == expected, 'ordering mismatch: ' + repr(trace))
        require(m.terminal == 'CANCELLED' and m.owner is None, 'terminal closure')
        count += 1
    return count


def check_fence(mutant=None):
    m = Gate(mutant)
    require(m.freeze() and m.grant(), 'handoff setup')
    require(not m.append('late', 'src', 0, 0, 7), 'stale owner accepted')
    require(m.out == [], 'stale output visible')


def check_idempotence(mutant=None):
    m = Gate(mutant)
    require(m.append('a', 'src', 0, 0, 7), 'initial append')
    require(m.freeze() and m.grant(), 'handoff setup')
    m.cancel()
    before = m.view()
    require(m.append('a', 'src', 0, 0, 7), 'lost acknowledgement not queryable')
    require(m.view() == before, 'duplicate commit installed again')
    require(m.freeze() and m.grant(), 'historical decision missing')
    require(m.view() == before, 'historical handoff reactivated request')
    require(not m.append('a', 'src', 0, 0, 8), 'same cid changed payload')
    require(m.view() == before, 'conflict mutated state')
    m.cancel()
    require(m.view() == before, 'duplicate cancellation changed terminal')


def check_parent(mutant=None):
    m = Gate(mutant)
    require(m.append('a', 'src', 0, 0, 7), 'initial append')
    require(not m.append('b', 'src', 0, 0, 8), 'old parent version accepted')
    require(m.out == [7], 'old parent changed history')


def check_ranks(mutant=None):
    for ranks, group in [((0,), 4), ((), 4), ((0, 1, 2), 4),
                         ((0, 0, 1), 4), ((0, 1), 3)]:
        m = Gate(mutant)
        require(not m.append('a', 'src', 0, 0, 7, ranks, group),
                'partial, duplicated, foreign rank or old group accepted')
        require(m.out == [], 'invalid group published')


def check_ready(mutant=None):
    m = Gate(mutant)
    require(m.freeze(), 'freeze setup')
    before = m.view()
    require(not m.grant(families={('KV', 0)}), 'partial target granted')
    require(m.view() == before, 'partial target changed authority')
    require(not m.grant(snapshot_version=99), 'wrong snapshot granted')
    require(m.grant(), 'complete target rejected')


def h_reference(xs):
    # Closed product-sum expansion, independent of the replay recurrence.
    return sum((2 * x + 1) * 3 ** (len(xs) - 1 - i) for i, x in enumerate(xs))


def replay(xs, c, p, checkpoint, mutant=None):
    h = checkpoint
    rng_cursor = 17
    emitted = []
    for t in range(c, p):
        x = xs[t]
        if mutant == 'resample':
            x = (x + 1) % 5
            rng_cursor += 1
            emitted.append(x)
        h = 3 * h + 2 * x + 1
    return h, rng_cursor, tuple(emitted)


def choose_start(families, p, records, budget, need_last_logits=False):
    # need_last_logits means logits are absent at p=A; an earlier start is required.
    top = p - int(need_last_logits)
    for c in range(top, -1, -1):
        if not all(c in positions for positions in families.values()):
            continue
        if c + 1 > budget:
            continue
        if not all(t in records for t in range(c, p)):
            continue
        return c
    return None


def check_start(c, families, xs):
    require(c is not None and all(c in d for d in families.values()),
            'non-common checkpoint')
    require(families['KV'][c] == tuple(xs[:c]), 'wrong KV prefix')
    require(families['REC'][c] == h_reference(xs[:c]), 'wrong recurrent position')


def check_replay(mutant=None):
    xs = DATA['tokens']
    count = 0
    for p in range(len(xs) + 1):
        for c in range(p + 1):
            result = replay(xs, c, p, h_reference(xs[:c]), mutant)
            require(result == (h_reference(xs[:p]), 17, ()),
                    'known-token replay changed state, RNG or output')
            count += 1
    return count


def check_recovery_points():
    xs = DATA['tokens']
    families = {
        'KV': {p: tuple(xs[:p]) for p in (0, 2, 4)},
        'REC': {p: h_reference(xs[:p]) for p in (0, 3, 4)},
    }
    records = dict(enumerate(xs))
    require(choose_start(families, 4, records, 5) == 4, 'direct recovery')
    require(choose_start(families, 4, records, 4) == 0, 'non-interval intersection')
    require(choose_start(families, 4, records, 0) is None, 'capacity rejection')
    require(choose_start(families, 4, {}, 5) == 4, 'direct path needs no replay')
    extended = {name: dict(values) for name, values in families.items()}
    extended['REC'][2] = h_reference(xs[:2])
    c = choose_start(extended, 4, records, 5, True)
    require(c == 2, 'missing last logits did not require earlier checkpoint')
    check_start(c, extended, xs)
    require(replay(xs, c, 4, extended['REC'][c]) == (h_reference(xs), 17, ()),
            'last-row regeneration changed confirmed history')
    require(choose_start(extended, 4, {0: xs[0]}, 5, True) is None,
            'missing replay input accepted')
    empty = {'KV': {0: ()}, 'REC': {0: 0}}
    require(choose_start(empty, 0, {}, 1) == 0, 'empty state recovery')
    require(choose_start(empty, 0, {}, 1, True) is None,
            'invented empty-prefix logits')
    no_initial = {'KV': {2: tuple(xs[:2])}, 'REC': {3: h_reference(xs[:3])}}
    require(choose_start(no_initial, 4, records, 5) is None,
            'invented initial checkpoint')
    for p in range(5):
        c = choose_start(extended, p, records, 5)
        if c is not None:
            check_start(c, extended, xs)
            require(replay(xs, c, p, extended['REC'][c])[0] == h_reference(xs[:p]),
                    'recovered state differs from closed form')


def check_bad_minimum():
    xs = DATA['tokens']
    asymmetric = {
        'KV': {p: tuple(xs[:p]) for p in (0, 2, 4)},
        'REC': {p: h_reference(xs[:p]) for p in (0, 3)},
    }
    bad = min(max(d) for d in asymmetric.values())
    check_start(bad, asymmetric, xs)


class Uses:
    def __init__(self, mutant=None):
        self.mutant = mutant
        self.owner = True
        self.active = set()
        self.ended = set()

    def complete(self, ticket):
        if ticket in self.active:
            self.active.remove(ticket)
            self.ended.add(ticket)
        elif self.mutant == 'duplicate' and self.active:
            self.active.pop()

    def timeout(self):
        if self.mutant == 'timeout':
            self.active.clear()

    def reusable(self):
        return not self.owner and not self.active


def check_uses(mutant=None):
    m = Uses(mutant)
    first = ('tx0', 'op1', 'slot', 0)
    second = ('tx0', 'op2', 'slot', 0)
    m.active.update((first, second))
    m.owner = False
    require(not m.reusable(), 'logical termination freed live accesses')
    m.complete(first)
    m.timeout()
    require(not m.reusable() and second in m.active, 'timeout freed live access')
    m.complete(first)
    require(not m.reusable() and second in m.active, 'duplicate settled another ticket')
    m.complete(second)
    require(m.reusable(), 'completed object cannot retire')
    fresh = ('tx1', 'op1', 'slot', 1)
    m.active.add(fresh)
    m.complete(first)
    require(not m.reusable() and fresh in m.active, 'old completion freed new generation')
    m.complete(fresh)
    require(m.reusable(), 'fresh ticket did not settle')


def check_costs():
    params = DATA['growth']
    D = Fraction(params['D_MiB'])
    B = Fraction(params['B_MiB_s'])
    g = Fraction(params['g_MiB_s'])
    remaining = D
    elapsed = Fraction(0)
    traffic = Fraction(0)
    for _ in range(params['online_stages']):
        duration = remaining / B
        traffic += remaining
        elapsed += duration
        remaining = g * duration
    pause = remaining / B
    traffic += remaining
    elapsed += pause
    require((traffic, elapsed, pause) == (1111, Fraction(1111, 100), Fraction(1, 100)),
            'migration event accounting')
    # Independent geometric expression, with explicit frozen final stage.
    ratio = g / B
    geometric = D * sum(ratio ** i for i in range(params['online_stages'] + 1))
    require(traffic == geometric, 'geometric reference differs')
    require(D * Fraction(120, 100) >= D, 'non-convergent growth counterexample')
    k = DATA['kv']
    token_bytes = 2 * k['layers'] * k['heads'] * k['width'] * k['element_bytes']
    logical = k['tokens'] * token_bytes
    require(token_bytes == 131072 and logical == 512 * 2 ** 20, 'KV dimensions')
    require(k['tokens'] * 8 == 32 * 2 ** 10, 'token byte accounting')
    require(Fraction(logical, 20 * 2 ** 30) == Fraction(25, 1000), 'copy service units')
    return {'migration_MiB': str(traffic), 'migration_seconds': str(elapsed),
            'final_pause_seconds': str(pause), 'KV_bytes_per_token': token_bytes}


def detect(name, check):
    try:
        check()
    except AssertionError:
        return name
    raise AssertionError('negative control escaped: ' + name)


def main():
    orderings = check_orderings()
    for check in (check_fence, check_idempotence, check_parent, check_ranks,
                  check_ready, check_recovery_points, check_uses):
        check()
    replay_cases = check_replay()
    cost = check_costs()
    detected = [
        detect('missing_fence', lambda: check_fence('fence')),
        detect('duplicate_install', lambda: check_idempotence('duplicate_install')),
        detect('old_parent', lambda: check_parent('parent')),
        detect('missing_rank_or_group_check', lambda: check_ranks('ranks')),
        detect('partial_target', lambda: check_ready('ready')),
        detect('minimum_of_frontiers', check_bad_minimum),
        detect('resample_during_replay', lambda: check_replay('resample')),
        detect('timeout_as_drain', lambda: check_uses('timeout')),
        detect('duplicate_completion', lambda: check_uses('duplicate')),
    ]
    input_bytes = json.dumps(DATA, sort_keys=True, separators=(',', ':')).encode('ascii')
    report = {
        'scope': 'finite CPU authority, toy recurrence, abstract access tickets and arithmetic only',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
        'python': list(sys.version_info[:3]),
        'ordered_traces_checked': orderings,
        'closed_form_replay_cases': replay_cases,
        'negative_controls_detected': detected,
        'cost_observations': cost,
        'unverified': ['GPU execution', 'real model logits', 'device or NIC visibility',
                       'physical abort and drain', 'distributed consensus and durability',
                       'network delivery', 'performance', 'Agent benefit'],
    }
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))


if __name__ == '__main__':
    main()
