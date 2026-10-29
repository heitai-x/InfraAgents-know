"""Independent first-phase transfer answer; does not import author checks."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
P, H, DK, DV, B = 3, 2, 3, 5, 2


def address(layout, physical, token, head, channel, dim):
    inner = (token*H+head)*dim+channel if layout == 'NHD' else (head*P+token)*dim+channel
    return physical*P*H*dim*B+inner*B


def logical_value(token, head, channel, value=False):
    return float(100*token+10*head+channel+(1000 if value else 0))


def make_pool(layout, dim, value=False):
    pool = bytearray(8*P*H*dim*B)
    for physical, token_range in ((5, range(3)), (2, range(3, 5)), (7, range(3, 6))):
        for token in token_range:
            for head in range(H):
                for channel in range(dim):
                    offset = address(layout, physical, token % P, head, channel, dim)
                    struct.pack_into('e', pool, offset, logical_value(token, head, channel, value))
    return pool


def dense_from_pages(pool, layout, table, length, dim):
    return [[[struct.unpack_from('e', pool, address(layout, table[t//P], t % P, h, c, dim))[0]
              for c in range(dim)] for h in range(H)] for t in range(length)]


@dataclass
class Page:
    generation: int = 1
    owners: set = field(default_factory=set)
    readers: set = field(default_factory=set)


class Lifecycle:
    def __init__(self):
        self.pages = {5: Page(owners={'B'}), 2: Page(owners={'B'}), 7: Page()}
        self.tickets = {}
        self.begin('R0', (5, 2))
        self.begin('R1', (5, 7))

    def begin(self, ticket, page_ids):
        assert ticket not in self.tickets
        self.tickets[ticket] = tuple((pid, self.pages[pid].generation) for pid in page_ids)
        for pid in page_ids:
            self.pages[pid].readers.add(ticket)

    def end(self, ticket):
        if ticket not in self.tickets:
            return False
        handles = self.tickets.pop(ticket)
        for pid, generation in handles:
            assert self.pages[pid].generation == generation
            self.pages[pid].readers.remove(ticket)
        return True

    def reusable(self, pid):
        return not self.pages[pid].owners and not self.pages[pid].readers

    def reuse(self, pid, owner):
        assert self.reusable(pid)
        self.pages[pid].generation += 1
        self.pages[pid].owners = {owner}


def main():
    offsets = {
        'NHD_K': address('NHD', 2, 1, 1, 2, DK),
        'NHD_V': address('NHD', 2, 1, 1, 4, DV),
        'HND_K': address('HND', 2, 1, 1, 2, DK),
        'HND_V': address('HND', 2, 1, 1, 4, DV)
    }
    assert offsets == {'NHD_K': 94, 'NHD_V': 158, 'HND_K': 100, 'HND_V': 168}
    for layout in ('NHD', 'HND'):
        for dim, is_value in ((DK, False), (DV, True)):
            pool = make_pool(layout, dim, is_value)
            for table, length in (((5, 2), 5), ((5, 7), 6)):
                reference = [[[logical_value(t, h, c, is_value) for c in range(dim)] for h in range(H)] for t in range(length)]
                assert dense_from_pages(pool, layout, table, length, dim) == reference
            opposite = 'HND' if layout == 'NHD' else 'NHD'
            assert dense_from_pages(pool, opposite, (5, 2), 5, dim) != dense_from_pages(pool, layout, (5, 2), 5, dim)

    initial = Lifecycle()
    after_cancel = {str(pid): {'owners': len(page.owners), 'pins': len(page.readers)} for pid, page in initial.pages.items()}
    assert after_cancel == {'5': {'owners': 1, 'pins': 2}, '2': {'owners': 1, 'pins': 1}, '7': {'owners': 0, 'pins': 1}}
    orders = []
    for order in (('R0', 'R1'), ('R1', 'R0')):
        model = Lifecycle()
        assert not any(model.reusable(pid) for pid in model.pages)
        for ticket in order:
            assert model.end(ticket)
            assert not model.end(ticket)
            assert not model.reusable(5) and not model.reusable(2)
            assert model.reusable(7) == ('R1' not in model.tickets)
        assert model.reusable(7)
        model.reuse(7, 'C')
        model.begin('R2', (7,))
        before = (model.pages[7].generation, model.pages[7].owners.copy(), model.pages[7].readers.copy())
        assert not model.end('R1')
        assert before == (model.pages[7].generation, model.pages[7].owners, model.pages[7].readers)
        orders.append(list(order))

    token_bytes = H*(DK+DV)*B
    cow = {'read_bytes': 2*token_bytes, 'write_bytes': 2*token_bytes, 'append_bytes': token_bytes}
    assert cow == {'read_bytes': 64, 'write_bytes': 64, 'append_bytes': 32}
    effective = 13*5*96*1024
    capacity = {}
    for page_tokens in (3, 8):
        entries = 13*((5+page_tokens-1)//page_tokens)
        allocated = entries*page_tokens*96*1024
        capacity['P'+str(page_tokens)] = {'effective_bytes': effective, 'allocated_bytes': allocated,
                                         'wasted_bytes': allocated-effective, 'table_entries': entries,
                                         'table_bytes': entries*4, 'preparation_us': entries*.3}
    assert capacity['P3']['allocated_bytes'] == 7667712 and capacity['P3']['wasted_bytes'] == 1277952
    assert capacity['P8']['allocated_bytes'] == 10223616 and capacity['P8']['wasted_bytes'] == 3833856
    documents = ['docs/evidence/parallel-mechanisms-20261027/tasks/paged-transfer.md', 'knowledge/algorithms/paged-kv-capacity.md']
    result = {
        'result': 'PASS', 'date': '2026-10-28', 'phase': 'initial_before_counterevidence',
        'offsets': offsets, 'after_cancel': after_cancel, 'cow': cow, 'capacity': capacity,
        'physical_page_pair_bytes': P*token_bytes, 'simultaneously_live_page_pairs': 3,
        'simultaneously_live_bytes': 3*P*token_bytes, 'completion_orders': orders,
        'hypotheses': [
            {'cause': 'Early reuse after cancellation ignores an in-flight reader', 'prediction': 'Delayed R1 and forced reuse amplify failures; retaining pages through final read suppresses this path'},
            {'cause': 'Mutable descriptor or current-row routing replaces the old snapshot', 'prediction': 'Can fail with pages retained; separate immutable descriptor/routing snapshots change behaviour'},
            {'cause': 'Duplicate old completion changes a recycled page generation', 'prediction': 'Injecting duplicate notifications corrupts new reader accounting unless ticket/generation checks isolate it'},
            {'cause': 'Reading beyond the valid tail', 'prediction': 'Poisoning the third tail slot can fail even without cancellation or reuse'}
        ],
        'documents': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in documents},
        'scope': 'Independent author-cross-reading CPU transfer exercise; private byte layout and set-based lifecycle construction. Shared parent context, no OS isolation, no fixed reasoning budget, no upstream code or author checks read. Not controlled knowledge benefit, real GPU safety or performance evidence.'
    }
    (HERE/'paged-answer.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'offsets': offsets, 'after_cancel': after_cancel, 'cow': cow, 'capacity': capacity}, ensure_ascii=False))


if __name__ == '__main__':
    main()
