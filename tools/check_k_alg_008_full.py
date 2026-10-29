from itertools import permutations


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def next_token(prefix):
    return (3 * sum(prefix) + len(prefix)) % 17 + 1


def serial_reference(prompt, limit):
    prefix = list(prompt) if prompt else [1]
    output = []
    for _ in range(limit):
        token = next_token(prefix)
        prefix.append(token)
        output.append(token)
    return output


class Model:
    # Unit reservations represent transient slots, not real KV pages or bytes.
    def __init__(self, slots=2, depth=2, chunk=2, fault=None):
        self.owner = [None] * slots
        self.epoch = [0] * slots
        self.depth = depth
        self.chunk = chunk
        self.fault = fault
        self.requests = {}
        self.generations = {}
        self.inflight = 0
        self.reserved = 0

    def new_generation(self, r, g):
        request = self.requests[r]
        request['active'] = g
        self.generations[r, g] = dict(
            phase='WAIT_SLOT', identity=None, pending=set(), counted=False,
            canceled=False, failed=False, business=False, need_next=False,
            samples=0, submitted=False, b=None, e=None, snapshot=None,
            candidate=None)

    def current(self, r):
        g = self.requests[r]['active']
        return self.generations[r, g] if g is not None else None

    def identity(self, r, g=0):
        return self.generations[r, g]['identity']

    def match(self, identity):
        if identity is None:
            return None
        r, g, s, epoch = identity
        record = self.generations.get((r, g))
        if record is None or record['identity'] != identity:
            return None
        if self.owner[s] != (r, g) or self.epoch[s] != epoch:
            return None
        if record['phase'] == 'RELEASED':
            return None
        return record

    def stop(self, r, reason):
        request = self.requests[r]
        if request['status'] == 'LIVE':
            request['status'] = reason
        request['active'] = None

    def release(self, r, g, record):
        if record['phase'] != 'DONE':
            return
        if record['pending'] and self.fault != 'early_release':
            return
        identity = record['identity']
        if identity is not None:
            _, _, s, epoch = identity
            require(self.owner[s] == (r, g), 'release lost ownership')
            require(self.epoch[s] == epoch, 'release mixed epoch')
            self.owner[s] = None
            self.epoch[s] += 1
            self.reserved -= 1
        if record['counted'] or self.fault == 'unconditional_decrement':
            self.inflight -= 1
            record['counted'] = False
        if self.fault == 'early_release':
            record['pending'].clear()
        record['phase'] = 'RELEASED'
        request = self.requests[r]
        if request['status'] == 'LIVE' and record['need_next']:
            self.new_generation(r, g + 1)

    def wake(self):
        for (r, g), record in list(self.generations.items()):
            self.release(r, g, record)
        for (r, g), record in list(self.generations.items()):
            request = self.requests[r]
            if record['phase'] == 'WAIT_SLOT' and request['status'] == 'LIVE':
                free = next((s for s, value in enumerate(self.owner)
                             if value is None), None)
                if free is not None:
                    require(request['k'] < len(request['x']), 'no executable input')
                    self.owner[free] = (r, g)
                    self.reserved += 1
                    record['identity'] = (r, g, free, self.epoch[free])
                    record['b'] = request['k']
                    record['e'] = min(len(request['x']), request['k'] + self.chunk)
                    record['snapshot'] = tuple(request['x'])
                    record['pending'] = {'prep'}
                    record['phase'] = 'PREPARING'
            if record['phase'] == 'READY' and self.inflight < self.depth:
                require(request['status'] == 'LIVE', 'submit stopped request')
                require(request['active'] == g, 'submit inactive generation')
                identity = record['identity']
                require(self.match(identity) is record, 'submit without ownership')
                if self.fault == 'empty_owner_submit':
                    if self.owner[identity[2]] is not None:
                        continue
                require(not record['pending'], 'submit before prepare finished')
                record['pending'].add('device')
                record['phase'] = 'SUBMITTED'
                record['submitted'] = True
                record['counted'] = True
                self.inflight += 1

    def check(self):
        counted = sum(record['counted'] for record in self.generations.values())
        require(self.inflight == counted, 'inflight accounting mismatch')
        require(0 <= self.inflight <= self.depth, 'inflight outside bound')
        require(self.reserved == sum(v is not None for v in self.owner),
                'reservation ownership mismatch')
        for s, key in enumerate(self.owner):
            if key is not None:
                record = self.generations[key]
                require(record['identity'] == (*key, s, self.epoch[s]),
                        'slot identity mismatch')
                require(record['phase'] != 'RELEASED', 'released owner')
        for record in self.generations.values():
            require(record['samples'] <= 1, 'duplicate sampling')
            if record['phase'] == 'RELEASED':
                require(not record['pending'], 'released live responsibility')
            if record['pending']:
                require(self.match(record['identity']) is record,
                        'pending use without protected slot')
        for request in self.requests.values():
            require(0 <= request['k'] <= len(request['x']), 'invalid KV boundary')

    def event(self, kind, *args, **payload):
        if kind == 'ARRIVE':
            r, prompt, limit = args
            if r not in self.requests:
                self.requests[r] = dict(x=list(prompt) if prompt else [1], k=0,
                    status='LIVE' if limit else 'DONE', active=None,
                    limit=limit, generated=0, rng=0, output=[])
                if limit:
                    self.new_generation(r, 0)
        elif kind == 'WAKE':
            self.wake()
        elif kind == 'CANCEL':
            r = args[0]
            self.stop(r, 'CANCELED')
            for (rr, _), record in self.generations.items():
                if rr == r and record['phase'] != 'RELEASED':
                    record['canceled'] = True
                    if record['phase'] in ('WAIT_SLOT', 'READY'):
                        record['phase'] = 'DONE'
        else:
            identity = args[0]
            record = self.match(identity)
            if record is not None:
                r, g, _, _ = identity
                request = self.requests[r]
                if kind == 'PREPARE_DONE':
                    if record['phase'] == 'PREPARING' and 'prep' in record['pending']:
                        record['pending'].remove('prep')
                        if payload.get('error', False):
                            record['failed'] = True
                            self.stop(r, 'ERROR')
                        record['phase'] = ('DONE' if record['canceled'] or
                            record['failed'] else 'READY')
                elif kind == 'DEVICE_DONE':
                    if record['phase'] == 'SUBMITTED' and 'device' in record['pending']:
                        if not (self.fault == 'cancel_rejects_done' and
                                record['canceled']):
                            record['pending'].remove('device')
                            record['phase'] = 'DONE'
                            layers = payload.get('layers', (True, True, True))
                            valid = len(layers) == 3 and all(layers)
                            if payload.get('error', False) or not valid:
                                record['failed'] = True
                                self.stop(r, 'ERROR')
                            elif request['status'] == 'LIVE' and request['active'] == g:
                                require(request['k'] == record['b'], 'wrong KV origin')
                                require(tuple(request['x']) == record['snapshot'],
                                        'input changed while in flight')
                                request['k'] = record['e']
                                if record['e'] < len(request['x']):
                                    record['need_next'] = True
                                else:
                                    record['candidate'] = payload.get(
                                        'candidate', next_token(record['snapshot']))
                                    record['pending'].update(
                                        {'consumer:result', 'consumer:copy',
                                         'consumer:archive'})
                elif kind == 'CONSUME_DONE':
                    cid = args[1]
                    uid = 'consumer:' + cid
                    present = uid in record['pending']
                    repeated_result = (self.fault == 'duplicate_result' and
                        cid == 'result' and record['business'])
                    if present or repeated_result:
                        record['pending'].discard(uid)
                        if cid == 'result':
                            if payload.get('error', False):
                                record['failed'] = True
                                self.stop(r, 'ERROR')
                            elif (request['status'] == 'LIVE' and
                                  request['active'] == g and
                                  not record['canceled'] and not record['failed'] and
                                  (not record['business'] or repeated_result)):
                                require(record['phase'] == 'DONE', 'sample before done')
                                require(request['k'] == record['e'], 'sample before KV')
                                record['business'] = True
                                record['samples'] += 1
                                request['rng'] += 1
                                request['generated'] += 1
                                token = record['candidate']
                                request['x'].append(token)
                                if token != 0:
                                    request['output'].append(token)
                                if token == 0 or request['generated'] >= request['limit']:
                                    self.stop(r, 'DONE')
                                else:
                                    record['need_next'] = True
                    elif self.fault == 'anonymous_count' and record['pending']:
                        # Wrong: a duplicate completion consumes another identity.
                        record['pending'].remove(sorted(record['pending'])[0])
                elif kind == 'USE_DONE':
                    uid = 'reader:' + args[1]
                    record['pending'].discard(uid)
                else:
                    raise AssertionError('unknown event ' + kind)
        self.check()

    def register_reader(self, identity, name):
        record = self.match(identity)
        require(record is not None, 'reader without ownership')
        require(record['phase'] == 'SUBMITTED', 'reader registration phase')
        # Registration precedes the conceptual reader start, under the owner.
        record['pending'].add('reader:' + name)
        self.check()


def start(m, r='A', prompt=(2, 3), limit=3):
    m.event('ARRIVE', r, prompt, limit)
    m.event('WAKE')
    record = m.current(r)
    require(record is not None and record['phase'] == 'PREPARING', 'arrival stuck')
    identity = record['identity']
    m.event('PREPARE_DONE', identity)
    m.event('WAKE')
    require(record['phase'] == 'SUBMITTED', 'READY submission stuck')
    return identity


def finish_consumers(m, identity, order=('result', 'copy', 'archive')):
    for cid in order:
        m.event('CONSUME_DONE', identity, cid)


def progression(fault=None, order=('result', 'copy', 'archive')):
    m = Model(chunk=1, fault=fault)
    m.event('ARRIVE', 'A', (2, 3, 4), 3)
    for _ in range(20):
        m.event('WAKE')
        for record in list(m.generations.values()):
            identity = record['identity']
            if record['phase'] == 'PREPARING':
                m.event('PREPARE_DONE', identity)
        m.event('WAKE')
        for record in list(m.generations.values()):
            identity = record['identity']
            if record['phase'] == 'SUBMITTED':
                m.event('DEVICE_DONE', identity)
                before = (m.requests['A']['k'], m.requests['A']['rng'])
                m.event('DEVICE_DONE', identity)
                require(before == (m.requests['A']['k'], m.requests['A']['rng']),
                        'duplicate device completion changed state')
                finish_consumers(m, identity, order)
                m.event('CONSUME_DONE', identity, 'result')
                m.event('CONSUME_DONE', identity, 'copy')
        m.event('WAKE')
        if m.requests['A']['status'] == 'DONE' and not any(m.owner):
            break
    require(m.requests['A']['output'] == serial_reference((2, 3, 4), 3),
            'output differs from independent serial reference')
    require(m.requests['A']['rng'] == 3, 'RNG advanced wrong number of times')
    require(m.requests['A']['k'] == 5, 'sampled token incorrectly acquired KV')
    require(m.inflight == 0 and m.reserved == 0, 'progression did not drain')
    require(len(m.generations) == 5, 'partial prefill or successor count wrong')
    return m


def cancellation(stage, fault=None):
    m = Model(slots=1, depth=1, fault=fault)
    m.event('ARRIVE', 'A', (2,), 2)
    identity = None
    if stage != 'WAIT_SLOT':
        m.event('WAKE')
        identity = m.identity('A')
    if stage in ('READY', 'SUBMITTED', 'DONE'):
        m.event('PREPARE_DONE', identity)
    if stage in ('SUBMITTED', 'DONE'):
        m.event('WAKE')
    if stage == 'DONE':
        m.event('DEVICE_DONE', identity)
    m.event('CANCEL', 'A')
    m.event('CANCEL', 'A')
    if stage == 'PREPARING':
        m.event('WAKE')
        require(m.reserved == 1, 'canceled preparation released early')
        m.event('PREPARE_DONE', identity)
    elif stage == 'SUBMITTED':
        m.event('WAKE')
        require(m.inflight == 1, 'cancel removed execution responsibility')
        m.event('DEVICE_DONE', identity)
    elif stage == 'DONE':
        finish_consumers(m, identity)
    m.event('WAKE')
    m.event('WAKE')
    require(m.requests['A']['output'] == [], 'canceled output committed')
    require(m.requests['A']['rng'] == 0, 'canceled sampling committed RNG')
    require(m.inflight == 0 and m.reserved == 0, 'cancel failed to drain')
    require(len(m.generations) == 1, 'canceled request created successor')


def reader_and_error(fault=None):
    m = Model(slots=1, depth=1, fault=fault)
    identity = start(m)
    m.register_reader(identity, 'cpu-transfer-source')
    m.event('DEVICE_DONE', identity, error=True)
    m.event('WAKE')
    require(m.owner[0] == ('A', 0), 'slot reused before external last use')
    require(m.inflight == 1, 'error discarded reader protection')
    require(m.requests['A']['output'] == [], 'failed device published output')
    require(not any(x.startswith('consumer:') for x in
                    m.generations['A', 0]['pending']), 'ghost failure consumer')
    m.event('USE_DONE', identity, 'cpu-transfer-source')
    m.event('USE_DONE', identity, 'cpu-transfer-source')
    m.event('WAKE')
    require(m.inflight == 0 and m.reserved == 0, 'error cleanup leaked')
    m.event('ARRIVE', 'B', (9,), 1)
    m.event('WAKE')
    new_identity = m.identity('B')
    require(new_identity[3] == identity[3] + 1, 'epoch did not advance')
    before = (tuple(m.requests['B']['x']), m.reserved, m.inflight)
    m.event('DEVICE_DONE', identity)
    m.event('CONSUME_DONE', identity, 'result')
    m.event('USE_DONE', identity, 'cpu-transfer-source')
    require(before == (tuple(m.requests['B']['x']), m.reserved, m.inflight),
            'stale completion mutated new owner')


def identity_consumers(fault=None):
    m = Model(fault=fault)
    identity = start(m, limit=3)
    m.event('DEVICE_DONE', identity)
    m.event('CONSUME_DONE', identity, 'copy')
    remaining = set(m.generations['A', 0]['pending'])
    m.event('CONSUME_DONE', identity, 'copy')
    require(m.generations['A', 0]['pending'] == remaining,
            'duplicate consumer removed another responsibility')
    m.event('CONSUME_DONE', identity, 'result')
    output = list(m.requests['A']['output'])
    rng = m.requests['A']['rng']
    m.event('CONSUME_DONE', identity, 'result')
    require(m.requests['A']['output'] == output and m.requests['A']['rng'] == rng,
            'duplicate result committed twice')
    m.event('WAKE')
    require(m.generations['A', 0]['phase'] == 'DONE', 'archive protection lost')
    m.event('CONSUME_DONE', identity, 'archive')
    m.event('WAKE')
    require(m.generations['A', 1]['phase'] == 'PREPARING', 'successor not woken')


def boundaries():
    m = Model()
    m.event('WAKE')
    m.event('ARRIVE', 'zero', (), 0)
    m.event('WAKE')
    require(not m.generations and not any(m.owner), 'zero work created execution')
    identity = start(m, r='bos', prompt=(), limit=1)
    require(m.requests['bos']['x'] == [1], 'empty input BOS policy missing')
    m.event('DEVICE_DONE', identity, candidate=0)
    finish_consumers(m, identity)
    m.event('WAKE')
    require(m.requests['bos']['output'] == [] and
            m.requests['bos']['status'] == 'DONE', 'EOS did not stop')
    require(m.requests['bos']['rng'] == 1, 'EOS sampled more than once')
    for preparation in (True, False):
        f = Model()
        f.event('ARRIVE', 'F', (3,), 1)
        f.event('WAKE')
        identity = f.identity('F')
        f.event('PREPARE_DONE', identity, error=preparation)
        if not preparation:
            f.event('WAKE')
            f.event('DEVICE_DONE', identity, layers=(True, False, True))
        f.event('WAKE')
        require(f.requests['F']['status'] == 'ERROR', 'incomplete producer accepted')
        require(f.requests['F']['k'] == 0 and f.requests['F']['rng'] == 0,
                'failed production advanced confirmed state')
        require(f.reserved == 0 and f.inflight == 0, 'failure leaked reservation')
    f = Model()
    identity = start(f, limit=2)
    f.event('DEVICE_DONE', identity)
    f.event('CONSUME_DONE', identity, 'result', error=True)
    f.event('WAKE')
    require(f.reserved == 1, 'consumer failure erased other readers')
    finish_consumers(f, identity, ('archive', 'copy'))
    f.event('WAKE')
    require(f.reserved == 0 and f.requests['A']['output'] == [],
            'failed result cleanup incorrect')


def concurrent_depth():
    m = Model(slots=2, depth=1)
    a = start(m, r='A', limit=1)
    m.event('ARRIVE', 'B', (8,), 1)
    m.event('WAKE')
    b = m.identity('B')
    m.event('PREPARE_DONE', b)
    m.event('WAKE')
    require(m.generations['B', 0]['phase'] == 'READY', 'depth bound bypassed')
    require(m.reserved == 2 and m.inflight == 1, 'prepare/execute overlap absent')
    m.event('DEVICE_DONE', a)
    finish_consumers(m, a)
    m.event('WAKE')
    require(m.generations['B', 0]['phase'] == 'SUBMITTED', 'credit wakeup missing')
    m.event('DEVICE_DONE', b)
    finish_consumers(m, b, ('archive', 'result', 'copy'))
    m.event('WAKE')
    require(m.requests['A']['output'] == serial_reference((2, 3), 1),
            'A result routing mismatch')
    require(m.requests['B']['output'] == serial_reference((8,), 1),
            'B result routing mismatch')
    require(m.reserved == 0 and m.inflight == 0, 'concurrent trace did not drain')


def detect(name, operation):
    try:
        operation()
    except AssertionError as error:
        print('negative control detected:', name, '|', str(error))
        return
    raise AssertionError('negative control escaped: ' + name)


def main():
    for order in permutations(('result', 'copy', 'archive')):
        progression(order=order)
    print('serial-reference agreement: chunked prefill, first output, three samples, six consumer orders')
    for stage in ('WAIT_SLOT', 'PREPARING', 'READY', 'SUBMITTED', 'DONE'):
        cancellation(stage)
    print('cancellation: five physical phases drained without output or RNG commit')
    identity_consumers()
    reader_and_error()
    boundaries()
    concurrent_depth()
    print('boundaries: duplicate identities, stale epoch, producer/result errors, last reader, EOS, zero work and depth wakeup')
    detect('READY incorrectly requires empty owner',
           lambda: progression(fault='empty_owner_submit'))
    detect('canceled execution rejects real completion',
           lambda: cancellation('SUBMITTED', fault='cancel_rejects_done'))
    detect('unsubmitted release decrements inflight',
           lambda: cancellation('READY', fault='unconditional_decrement'))
    detect('duplicate result samples again',
           lambda: identity_consumers(fault='duplicate_result'))
    detect('duplicate consumer consumes another identity',
           lambda: identity_consumers(fault='anonymous_count'))
    detect('release before independent last use',
           lambda: reader_and_error(fault='early_release'))
    print('scope: deterministic CPU event model and toy function only; no GPU, DMA, graph, real KV allocator or performance measurements')


if __name__ == '__main__':
    main()
