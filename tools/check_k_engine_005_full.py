#!/usr/bin/env python3
"""K-ENGINE-005: deterministic, offline, read-only CPU model.

No project/upstream imports, files, subprocesses, credentials or network.
Synthetic integer layers test positions and state chains, not GPU numerics.
The script reports results only when actually executed.
"""
from dataclasses import dataclass, field
import sys

L = 3
V = 11
ALL_LAYERS = frozenset(range(L))


def initial_state(prompt, seed):
    counts = [0] * V
    for token in prompt:
        counts[token] += 1
    return (seed, sum(t + 1 for t in prompt) % 5, tuple(counts))


def incremental_row(token, pos, prefix):
    """One synthetic causal input row; each layer uses lower-layer h."""
    h = (token + 1) * 7 + pos
    rows = []
    for layer in range(L):
        assert len(prefix[layer]) == pos
        key = (h * (layer + 2) + pos + 1) % 97
        value = (h + key + 3 * layer) % 101
        total = sum(v for _, v in prefix[layer]) + value
        h = (h + key + total + layer + 1) % 127
        rows.append((key, value))
    scores = tuple((h * (j + 1) + j * j + pos) % 53 + 1
                   for j in range(V))
    return tuple(rows), scores


def incremental_sample(scores, state):
    rng, grammar, counts = state
    for _ in range(1 + sum(scores) % 2):
        rng = (25173 * rng + 13849) % 65536
    weights = [(j, max(1, scores[j] // (1 + counts[j])))
               for j in range(V) if (j + grammar) % 3 != 0]
    draw = rng % sum(w for _, w in weights)
    cumulative = 0
    for token, weight in weights:
        cumulative += weight
        if draw < cumulative:
            updated = list(counts)
            updated[token] += 1
            return token, (rng, (grammar + token + 1) % 5, tuple(updated))
    raise AssertionError('nonempty categorical distribution was not consumed')


def serial_reference(prompt, count, seed):
    """Independent full-history recomputation and categorical selection.

    Does not call incremental_row, incremental_sample or Pipeline.
    Recomputes all input positions for each new output.
    """
    history = list(prompt)
    rng = seed
    grammar = 0
    frequencies = [0 for _ in range(V)]
    for token in prompt:
        grammar = (grammar + token + 1) % 5
        frequencies[token] += 1
    outputs, states, input_rows = [], [], []
    for _ in range(count):
        layers = [[] for _ in range(L)]
        last_hidden = None
        for pos in range(len(history)):
            hidden = 7 * (history[pos] + 1) + pos
            for layer in range(L):
                k = ((layer + 2) * hidden + (pos + 1)) % 97
                v = (hidden + k + layer * 3) % 101
                layers[layer].append((k, v))
                history_sum = 0
                for old_k, old_v in layers[layer]:
                    history_sum += old_v
                hidden = (hidden + k + history_sum + (layer + 1)) % 127
            last_hidden = hidden
        pos = len(history) - 1
        raw = [((j + 1) * last_hidden + j ** 2 + pos) % 53 + 1
               for j in range(V)]
        draws = 1 if sum(raw) % 2 == 0 else 2
        for _ in range(draws):
            rng = (rng * 25173 + 13849) & 65535
        allowed, masses = [], []
        for j in range(V):
            if (grammar + j) % 3:
                allowed.append(j)
                masses.append(max(1, raw[j] // (frequencies[j] + 1)))
        choice = rng % sum(masses)
        selected = None
        for j, mass in zip(allowed, masses):
            if choice < mass:
                selected = j
                break
            choice -= mass
        assert selected is not None
        frequencies[selected] += 1
        grammar = (grammar + selected + 1) % 5
        input_rows.append(tuple(layer[-1] for layer in layers))
        states.append((rng, grammar, tuple(frequencies)))
        outputs.append(selected)
        history.append(selected)
    return outputs, states, input_rows


@dataclass
class Region:
    owners: set = field(default_factory=set)
    users: set = field(default_factory=set)
    freed: bool = False


@dataclass
class Step:
    g: int
    pos: int
    path: tuple
    q_in: tuple
    prefix: tuple
    accesses: tuple
    physical: str = 'SUBMITTED'
    success: bool = False
    business: str = 'PENDING'
    ticket: object = None
    successor_snapshot: bool = False
    rows: tuple = ()


class Pipeline:
    """Host commit is atomic here; device and external lifetimes are explicit."""
    def __init__(self, prompt=(2, 4, 1), depth=2, limit=8, seed=19,
                 fault='', eos=None):
        if not prompt or depth < 1 or limit < 0:
            raise ValueError('normalized nonempty input, depth>=1, limit>=0')
        if any(t < 0 or t >= V for t in prompt):
            raise ValueError('token outside synthetic vocabulary')
        self.prompt = tuple(prompt)
        self.depth, self.limit, self.seed = depth, limit, seed
        self.fault, self.eos = fault, eos
        self.instance = 17
        self.initial = initial_state(self.prompt, seed)
        self.official_q = self.initial
        self.history = list(prompt)
        self.public_k = len(prompt) - 1
        self.ns = self.nc = 0
        self.live = True
        self.terminal = None
        self.steps = {}
        self.outputs, self.states, self.order = [], [], []
        self.regions = {'prefix': Region({'request'})}
        self.base = [[] for _ in range(L)]
        for pos, token in enumerate(self.prompt[:-1]):
            rows, _ = incremental_row(token, pos, self.base)
            for layer in range(L):
                self.base[layer].append(rows[layer])
        if limit == 0:
            self.terminate('LENGTH')

    def retire_tickets(self):
        for step in self.steps.values():
            if step.business == 'DISCARDED':
                step.ticket = None
            elif step.business == 'COMMITTED' and (
                    step.successor_snapshot or not self.live):
                step.ticket = None

    def submit(self):
        if not self.live or self.ns - self.nc >= self.depth:
            return False
        g = self.ns
        if g == 0:
            path, q_in = self.prompt, self.initial
        else:
            prev = self.steps[g - 1]
            if prev.physical != 'ENDED' or not prev.success:
                return False
            if prev.business not in ('PENDING', 'COMMITTED'):
                return False
            if self.fault == 'phase_after_commit' and prev.business == 'COMMITTED':
                return False
            if prev.ticket is None:
                return False
            token, q_in = prev.ticket
            path = prev.path + (token,)
            if self.fault == 'rng_reset':
                q_in = self.initial
        prefix = []
        for layer in range(L):
            values = list(self.base[layer])
            for old in range(g):
                prior = self.steps[old]
                assert prior.physical == 'ENDED' and prior.success
                values.append(prior.rows[layer])
            prefix.append(tuple(values))
        name = 'kv:' + str(g)
        self.regions[name] = Region({'chain'})
        accesses = ('prefix',) + tuple('kv:' + str(i) for i in range(g + 1))
        uid = 'device:' + str(g)
        for region_name in accesses:
            region = self.regions[region_name]
            assert not region.freed
            region.users.add(uid)
        step = Step(g, len(self.prompt) - 1 + g, tuple(path), q_in,
                    tuple(prefix), accesses)
        self.steps[g] = step
        self.ns += 1
        if g:
            self.steps[g - 1].successor_snapshot = True
        self.retire_tickets()
        return True

    def finish(self, g, success=True, last_access=True, layers=None,
               instance=17):
        if instance != self.instance or g not in self.steps:
            return False
        step = self.steps[g]
        if step.physical != 'SUBMITTED':
            return False
        if not last_access:
            self.terminate('ERROR_UNKNOWN_END')
            return True
        coverage = ALL_LAYERS if layers is None else frozenset(layers)
        valid = success and coverage == ALL_LAYERS
        if valid:
            pos = step.pos + (1 if self.fault == 'kv_shift' else 0)
            if self.fault == 'kv_shift':
                # Faulty write position: retain valid input history but use
                # wrong position arithmetic, rather than failing a precheck.
                h = (step.path[-1] + 1) * 7 + pos
                rows = []
                for layer in range(L):
                    key = (h * (layer + 2) + pos + 1) % 97
                    value = (h + key + 3 * layer) % 101
                    total = sum(v for _, v in step.prefix[layer]) + value
                    h = (h + key + total + layer + 1) % 127
                    rows.append((key, value))
                scores = tuple((h * (j + 1) + j * j + pos) % 53 + 1
                               for j in range(V))
                step.rows = tuple(rows)
            else:
                step.rows, scores = incremental_row(step.path[-1], pos,
                                                     step.prefix)
            step.ticket = incremental_sample(scores, step.q_in)
        uid = 'device:' + str(g)
        for name in step.accesses:
            self.regions[name].users.remove(uid)
        step.physical, step.success = 'ENDED', valid
        if not valid:
            self.terminate('ERROR_RESULT')
        self.settle()
        return True

    def commit(self, g=None, stop=False, reject=False):
        g = self.nc if g is None else g
        if not self.live or g not in self.steps:
            return False
        step = self.steps[g]
        if (step.physical != 'ENDED' or not step.success or
                step.business != 'PENDING' or step.ticket is None):
            return False
        if self.fault != 'unordered':
            if g != self.nc:
                return False
            if (step.path != tuple(self.history) or
                    step.q_in != self.official_q or
                    step.pos != len(self.prompt) - 1 + self.nc):
                self.terminate('ERROR_CHAIN')
                return False
        if reject:
            self.terminate('HOST_REJECT')
            return False
        token, q_out = step.ticket
        self.outputs.append(token)
        self.states.append(q_out)
        self.history.append(token)
        self.official_q = q_out
        self.public_k = step.pos + 1
        self.order.append(g)
        step.business = 'COMMITTED'
        region = self.regions['kv:' + str(g)]
        region.owners.remove('chain')
        region.owners.add('request')
        self.nc += 1
        if stop or token == self.eos:
            self.terminate('EOS')
        elif self.nc >= self.limit:
            self.terminate('LENGTH')
        self.settle()
        return True

    def terminate(self, reason):
        if not self.live:
            return
        self.live = False
        self.terminal = reason
        for step in self.steps.values():
            if step.business == 'PENDING':
                step.business = 'DISCARD_PENDING'
        for region in self.regions.values():
            region.owners.discard('request')
            region.owners.discard('chain')
            if self.fault == 'erase_users':
                region.users.clear()
        self.settle()

    def settle(self):
        for g, step in sorted(self.steps.items()):
            if step.business != 'DISCARD_PENDING' or step.physical != 'ENDED':
                continue
            if (self.fault == 'commit_gate_drain' and g > 0 and
                    self.steps[g - 1].business != 'COMMITTED'):
                continue
            step.business = 'DISCARDED'
        self.retire_tickets()
        for region in self.regions.values():
            if not region.owners and not region.users:
                region.freed = True

    def begin_reader(self, name, uid):
        region = self.regions[name]
        assert not region.freed and uid not in region.users
        region.users.add(uid)

    def end_reader(self, name, uid):
        region = self.regions[name]
        if uid not in region.users:
            return False
        region.users.remove(uid)
        self.settle()
        return True

    def drained(self):
        return (not self.live and
                all(s.physical == 'ENDED' and
                    s.business in ('COMMITTED', 'DISCARDED')
                    for s in self.steps.values()) and
                all(not r.users and not r.owners and r.freed
                    for r in self.regions.values()))


def audit(p):
    """Reconstruct expected accesses independently from physical states."""
    assert p.order == list(range(p.nc)), 'commits must form a prefix'
    assert p.history == list(p.prompt) + p.outputs
    assert p.public_k == len(p.prompt) - 1 + p.nc, 'KV/token offset'
    assert 0 <= p.ns - p.nc <= p.depth
    for name, region in p.regions.items():
        expected = set()
        for g, step in p.steps.items():
            if step.physical == 'SUBMITTED' and name in step.accesses:
                expected.add('device:' + str(g))
        actual = {u for u in region.users if u.startswith('device:')}
        assert actual == expected, 'unended device access must retain its uid'
        if region.freed:
            assert not region.users and not region.owners, 'premature reuse'
        if not p.live:
            assert 'request' not in region.owners and 'chain' not in region.owners
    for g, step in p.steps.items():
        assert step.pos == len(p.prompt) - 1 + g
        assert len(step.path) == step.pos + 1
        assert all(len(layer) == step.pos for layer in step.prefix)
        if step.business == 'COMMITTED':
            assert step.physical == 'ENDED' and step.success
        if step.business == 'DISCARDED':
            assert step.physical == 'ENDED' and step.ticket is None
        if not p.live:
            assert step.business != 'PENDING'


def run_pipeline(prompt, depth, limit, seed, policy):
    p = Pipeline(prompt, depth, limit, seed)
    audit(p)
    budget = 30 * (limit + depth + 1)
    for _ in range(budget):
        if p.drained():
            return p
        changed = False
        if p.live:
            actions = ('commit', 'submit', 'finish') if policy == 'commit_first' else (
                'submit', 'finish', 'commit')
            for action in actions:
                if action == 'commit':
                    changed = p.commit()
                elif action == 'submit':
                    changed = p.submit()
                else:
                    waiting = [g for g, s in p.steps.items()
                               if s.physical == 'SUBMITTED']
                    changed = bool(waiting) and p.finish(min(waiting))
                if changed:
                    break
        else:
            waiting = [g for g, s in p.steps.items() if s.physical == 'SUBMITTED']
            if waiting:
                changed = p.finish(min(waiting))
            else:
                p.settle()
                changed = p.drained()
        audit(p)
        assert changed, 'finite ready workload has stalled'
    raise AssertionError('finite workload exceeded progress budget')


def check_reference_trajectories():
    cases = 0
    for prompt in ((0,), (2, 4), (10, 1, 7, 3)):
        for seed in (0, 19, 65535):
            for count in (0, 1, 2, 7, 11):
                expected, states, rows = serial_reference(prompt, count, seed)
                for depth in (1, 2, 4):
                    for policy in ('commit_first', 'fill_first'):
                        p = run_pipeline(prompt, depth, count, seed, policy)
                        assert p.outputs == expected, 'serial token reference'
                        assert p.states == states, 'full sampling-state reference'
                        for g in range(count):
                            assert p.steps[g].rows == rows[g], 'per-layer KV reference'
                        discarded = {g for g, s in p.steps.items()
                                     if s.business == 'DISCARDED'}
                        assert discarded == set(range(count, p.ns))
                        assert p.drained()
                        cases += 1
    print('OBS serial full-history reference cases=' + str(cases) +
          '; depths=1,2,4; outputs through 11; integer causal layers only')


def check_commit_before_successor():
    p = Pipeline(depth=2, limit=9)
    for g in range(9):
        assert p.submit(), 'committed predecessor must permit its successor'
        if g:
            assert p.steps[g - 1].physical == 'ENDED'
            assert p.steps[g - 1].business == 'COMMITTED'
            assert p.steps[g - 1].ticket is None, 'successor copied retired ticket'
        assert p.finish(g)
        assert p.commit(g)
        audit(p)
    assert p.nc == 9 and p.drained()
    assert not p.submit()
    print('OBS commit-before-successor: 9 commits at depth 2; physical completion persists')


def check_order_and_capacity():
    p = Pipeline(depth=2, limit=5)
    assert p.submit() and not p.submit()
    assert not p.finish(0, instance=16)
    assert p.steps[0].physical == 'SUBMITTED'
    assert p.finish(0) and not p.finish(0)
    assert p.submit() and p.finish(1)
    assert not p.submit(), 'candidate window full'
    assert not p.commit(1), 'later complete candidate cannot commit first'
    audit(p)
    assert p.commit(0)
    assert p.submit(), 'commit returns an admission credit'
    audit(p)
    p.terminate('CANCEL')
    assert p.finish(2)
    assert p.drained()
    print('OBS capacity, request identity, duplicate completion and ordered commit checked')


def check_stop_and_drain():
    p = Pipeline(depth=2, limit=20)
    assert p.submit() and p.finish(0) and p.submit()
    p.begin_reader('kv:0', 'save')
    assert p.commit(0, stop=True)
    assert p.terminal == 'EOS' and p.nc == 1
    assert p.steps[1].business == 'DISCARD_PENDING'
    assert p.steps[1].physical == 'SUBMITTED'
    assert 'device:1' in p.regions['kv:0'].users
    assert 'device:1' in p.regions['kv:1'].users
    assert not p.regions['kv:0'].freed and not p.regions['kv:1'].freed
    assert not p.drained() and not p.commit(1) and not p.submit()
    audit(p)
    assert p.finish(1)
    assert p.steps[1].business == 'DISCARDED' and p.steps[1].ticket is None
    assert p.regions['kv:1'].freed
    assert not p.regions['kv:0'].freed, 'independent save still holds source'
    assert not p.drained()
    assert p.end_reader('kv:0', 'save')
    assert not p.end_reader('kv:0', 'save')
    assert p.drained() and p.ns - p.nc == 1
    audit(p)

    # Multiple finished suffix candidates must settle despite no COMMITTED parent.
    rejected = Pipeline(depth=4, limit=20)
    for g in range(3):
        assert rejected.submit() and rejected.finish(g)
    assert rejected.submit()
    assert not rejected.commit(0, reject=True)
    assert rejected.nc == 0
    assert all(rejected.steps[g].business == 'DISCARDED' for g in range(3))
    assert rejected.steps[3].business == 'DISCARD_PENDING'
    assert not rejected.drained()
    audit(rejected)
    assert rejected.finish(3)
    assert all(s.business == 'DISCARDED' for s in rejected.steps.values())
    assert rejected.drained()
    audit(rejected)

    for where in ('before_submit', 'running', 'finished_suffix'):
        canceled = Pipeline(depth=3, limit=20)
        if where != 'before_submit':
            assert canceled.submit()
        if where == 'finished_suffix':
            assert canceled.finish(0) and canceled.submit() and canceled.finish(1)
        canceled.terminate('CANCEL')
        canceled.terminate('ERROR_LATE')
        assert canceled.terminal == 'CANCEL'
        audit(canceled)
        for g, step in canceled.steps.items():
            if step.physical == 'SUBMITTED':
                assert not canceled.drained()
                assert canceled.finish(g)
        assert canceled.drained() and canceled.outputs == []
        assert all(s.business == 'DISCARDED' for s in canceled.steps.values())
    print('OBS EOS, host rejection and cancel: suffix discard, retained in-flight uid, independent reader, full drain')


def check_incomplete_evidence():
    p = Pipeline()
    assert p.submit()
    assert p.finish(0, last_access=False)
    assert p.steps[0].physical == 'SUBMITTED'
    assert p.steps[0].business == 'DISCARD_PENDING'
    assert p.regions['kv:0'].users == {'device:0'}
    assert not p.drained()
    audit(p)
    assert p.finish(0, success=False, last_access=True)
    assert p.drained() and p.outputs == []
    audit(p)

    missing_layer = Pipeline()
    assert missing_layer.submit()
    assert missing_layer.finish(0, layers=(0, 1))
    assert not missing_layer.steps[0].success
    assert missing_layer.steps[0].business == 'DISCARDED'
    assert missing_layer.outputs == [] and missing_layer.drained()
    audit(missing_layer)
    print('OBS missing end evidence retains access; ended but incomplete-layer result is discarded')


def resource_clock(n, d, h, depth):
    """Integer-time simulation, independent of the closed-form expression."""
    if n == 0:
        return 0
    submitted = committed = 0
    done = set()
    device = host = None
    for tick in range(n * (d + h) + 2):
        if host is not None and host[1] == tick:
            assert host[0] == committed
            committed += 1
            host = None
        if device is not None and device[1] == tick:
            done.add(device[0])
            device = None
        if committed == n:
            return tick
        if host is None and committed in done:
            host = (committed, tick + h)
        if (device is None and submitted < n and
                submitted - committed < depth and
                (submitted == 0 or submitted - 1 in done)):
            device = (submitted, tick + d)
            submitted += 1
    raise AssertionError('time simulation stalled')


def check_times():
    cases = 0
    for n in range(8):
        for d in range(1, 6):
            for h in range(1, 6):
                expected = 0 if n == 0 else d + h + (n - 1) * max(d, h)
                assert resource_clock(n, d, h, 2) == expected
                assert resource_clock(n, d, h, 1) == n * (d + h)
                cases += 1
    assert resource_clock(3, 2, 5, 2) == 17
    print('OBS integer resource-clock cases=' + str(cases) +
          '; constant independent stages, depth 1 serial and depth 2 formula')


@dataclass
class Guard:
    epoch: int
    open: bool = True
    released: bool = False
    outstanding: set = field(default_factory=set)
    completed: set = field(default_factory=set)
    error: str = ''

    def begin(self, cid):
        if (not self.open or self.released or cid in self.outstanding or
                cid in self.completed):
            return False
        self.outstanding.add(cid)
        return True

    def maybe_release(self):
        if not self.open and not self.outstanding:
            self.released = True

    def seal(self):
        if not self.open:
            return False
        self.open = False
        self.maybe_release()
        return True

    def complete(self, epoch, cid, ok=True, ended=True):
        if epoch != self.epoch or self.released or cid not in self.outstanding:
            return False
        if not ok and not self.error:
            self.error = 'first transfer failure'
        if not ended:
            return False
        self.outstanding.remove(cid)
        self.completed.add(cid)
        self.maybe_release()
        return True


class FirstCompletionGuard(Guard):
    def maybe_release(self):
        if self.completed:
            self.released = True


def audit_guard(g):
    assert not (g.released and (g.open or g.outstanding)), 'guard released with future/actual reader'
    assert not (g.outstanding & g.completed)


def check_guards():
    g = Guard(7)
    assert g.begin('send') and not g.begin('send')
    assert g.complete(7, 'send')
    assert not g.released, 'open zero-count interval still protected'
    assert g.begin('save') and g.seal()
    assert not g.begin('late')
    assert not g.complete(7, 'send')
    assert not g.complete(6, 'save')
    assert not g.complete(7, 'unknown')
    assert not g.complete(7, 'save', ok=False, ended=False)
    assert g.outstanding == {'save'} and not g.released
    first_error = g.error
    audit_guard(g)
    assert g.complete(7, 'save', ok=False, ended=True)
    assert g.error == first_error and g.released
    audit_guard(g)
    empty = Guard(8)
    assert empty.seal() and empty.released
    audit_guard(empty)
    print('OBS sealed guard: zero-count open interval, multiple readers, duplicates, old epoch, failed-but-ended distinction')


def require_detection(label, experiment):
    try:
        experiment()
    except AssertionError as exc:
        print('NEGATIVE_DETECTED ' + label + ': ' + str(exc))
        return
    raise AssertionError('negative control escaped detection: ' + label)


def negative_phase():
    p = Pipeline(fault='phase_after_commit')
    assert p.submit() and p.finish(0) and p.commit(0)
    assert p.submit(), 'mutant loses progress after predecessor COMMITTED'


def negative_drain_gate():
    p = Pipeline(depth=3, fault='commit_gate_drain')
    for g in range(3):
        assert p.submit() and p.finish(g)
    p.terminate('HOST_REJECT')
    assert p.drained(), 'mutant leaves discarded-parent suffix unsettled'


def negative_erased_users():
    p = Pipeline(fault='erase_users')
    assert p.submit() and p.finish(0) and p.submit()
    assert p.commit(0, stop=True)
    audit(p)


def negative_rng():
    p = Pipeline(fault='rng_reset')
    expected, states, rows = serial_reference(p.prompt, 2, p.seed)
    assert p.submit() and p.finish(0) and p.submit() and p.finish(1)
    assert p.steps[1].ticket[1] == states[1], 'mutant resets provisional sampling state'


def negative_kv_position():
    p = Pipeline(fault='kv_shift')
    expected, states, rows = serial_reference(p.prompt, 1, p.seed)
    assert p.submit() and p.finish(0)
    assert p.steps[0].rows == rows[0], 'mutant writes KV for wrong input position'


def negative_order():
    p = Pipeline(fault='unordered')
    assert p.submit() and p.finish(0) and p.submit() and p.finish(1)
    assert p.commit(1)
    audit(p)


def negative_guard():
    g = FirstCompletionGuard(3)
    assert g.begin('send') and g.begin('save') and g.seal()
    assert g.complete(3, 'send')
    audit_guard(g)


def main():
    print('SCOPE CPU synthetic state/dataflow/time models; no CUDA, TensorRT-LLM, network or performance measurements')
    check_reference_trajectories()
    check_commit_before_successor()
    check_order_and_capacity()
    check_stop_and_drain()
    check_incomplete_evidence()
    check_times()
    check_guards()
    for label, experiment in (
            ('predecessor phase conflation', negative_phase),
            ('commit-gated drain', negative_drain_gate),
            ('premature responsibility deletion', negative_erased_users),
            ('provisional RNG reset', negative_rng),
            ('KV input-position shift', negative_kv_position),
            ('out-of-order commit', negative_order),
            ('first-consumer guard release', negative_guard)):
        require_detection(label, experiment)
    print('CHECK_OK scoped CPU invariants and independent references only')
    return 0


if __name__ == '__main__':
    sys.exit(main())
