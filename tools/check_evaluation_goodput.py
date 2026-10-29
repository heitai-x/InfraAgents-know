#!/usr/bin/env python3
"""Synthetic H/J certificate integration: event machine vs declarative facts."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json

COMMON_FIXTURE = [{'certificate_version': 'v1',
  'content': ((10, 1), (20, 1), (30, 1)),
  'decision': (1, True),
  'material_ready_at': 30,
  'material_token_count': 3,
  'mode': 'TOKEN',
  'name': 'A',
  'output_version': 'v1',
  'planned': 0,
  'prefill_work': 20,
  'prompt': 100,
  'quality': (180, True),
  'quality_events': (),
  'ready': 0,
  'sent': 0,
  'terminal': (45, 'FINAL')},
 {'certificate_version': 'v1',
  'content': ((40, 1), (50, 1), (60, 1)),
  'decision': (31, True),
  'material_ready_at': 60,
  'material_token_count': 3,
  'mode': 'TOKEN',
  'name': 'B',
  'output_version': 'v1',
  'planned': 10,
  'prefill_work': 200,
  'prompt': 200,
  'quality': (75, True),
  'quality_events': (),
  'ready': 10,
  'sent': 30,
  'terminal': (65, 'FINAL')},
 {'certificate_version': 'v1',
  'content': ((50, 3),),
  'decision': (21, True),
  'material_ready_at': 50,
  'material_token_count': 3,
  'mode': 'CHUNK',
  'name': 'C',
  'output_version': 'v1',
  'planned': 20,
  'prefill_work': 100,
  'prompt': 300,
  'quality': (70, True),
  'quality_events': (),
  'ready': 20,
  'sent': 20,
  'terminal': (55, 'FINAL')},
 {'certificate_version': 'v1',
  'content': (),
  'decision': (35, False),
  'material_ready_at': None,
  'material_token_count': None,
  'mode': 'TOKEN',
  'name': 'D',
  'output_version': 'v1',
  'planned': 30,
  'prefill_work': 0,
  'prompt': 400,
  'quality': None,
  'quality_events': (),
  'ready': 30,
  'sent': 30,
  'terminal': (35, 'REJECT')},
 {'certificate_version': 'v1',
  'content': ((60, 1),),
  'decision': (41, True),
  'material_ready_at': None,
  'material_token_count': None,
  'mode': 'TOKEN',
  'name': 'E',
  'output_version': 'v1',
  'planned': 40,
  'prefill_work': 500,
  'prompt': 500,
  'quality': None,
  'quality_events': (),
  'ready': 40,
  'sent': 40,
  'terminal': (100, 'TIMEOUT')},
 {'certificate_version': 'v1',
  'content': ((70, 1), (80, 1), (90, 1)),
  'decision': (51, True),
  'material_ready_at': 90,
  'material_token_count': 3,
  'mode': 'TOKEN',
  'name': 'F',
  'output_version': 'v1',
  'planned': 50,
  'prefill_work': 600,
  'prompt': 600,
  'quality': None,
  'quality_events': (),
  'ready': 50,
  'sent': 50,
  'terminal': (170, 'FINAL')},
 {'certificate_version': 'v1',
  'content': (),
  'decision': (90, True),
  'material_ready_at': None,
  'material_token_count': None,
  'mode': 'TOKEN',
  'name': 'G',
  'output_version': 'v1',
  'planned': 60,
  'prefill_work': 0,
  'prompt': 700,
  'quality': None,
  'quality_events': (),
  'ready': 60,
  'sent': 60,
  'terminal': (80, 'CANCEL')},
 {'certificate_version': 'v1',
  'content': ((100, 1),),
  'decision': (71, True),
  'material_ready_at': 100,
  'material_token_count': 1,
  'mode': 'TOKEN',
  'name': 'H',
  'output_version': 'v1',
  'planned': 70,
  'prefill_work': 800,
  'prompt': 800,
  'quality': (120, True),
  'quality_events': (),
  'ready': 70,
  'sent': 70,
  'terminal': (110, 'FINAL')}]
COMMON_FIXTURE_SHA256 = '4b1d581c771b2a5bbfeac81999f25445f43ab76fcf5e486c8274e97ab80306ce'
RULE = 'task-quality-v1'


class CheckError(AssertionError):
    pass


def same(actual, expected, name):
    if actual != expected:
        raise CheckError(f'{name}: {actual!r} != {expected!r}')


def certificate_machine(row, J, final_time, last_client, fault):
    pending = []
    if row['quality'] is not None:
        pending.append(('primary', row['quality'][0], row['quality'][1], row['certificate_version'], RULE))
    pending.extend(tuple(value) for value in row['quality_events'])
    pending.sort(key=lambda event: (event[1], event[0]))
    verdict = None
    identities = {}
    conflict = False
    for identity, when, passed, version, rule in pending:
        if when > J and fault != 'include_future_certificate':
            continue
        if rule != RULE or version != row['output_version']:
            continue  # A separate output/rule view never supplies this view's PASS.
        key = (row['name'], version, rule, identity)
        payload = (when, passed)
        if key in identities:
            if identities[key] != payload:
                conflict = True
            continue
        identities[key] = payload
        if type(passed) is not bool or type(when) is not int:
            conflict = True
            continue
        material_ready = row['material_ready_at']
        if material_ready is not None and when < material_ready:
            conflict = True
        if fault == 'quality_requires_final' and final_time is not None and when < final_time:
            continue
        if fault == 'quality_requires_client_last' and last_client is not None and when < last_client:
            continue
        if verdict is not None and verdict != passed:
            if fault != 'last_verdict_wins':
                conflict = True
        verdict = passed
    if conflict:
        return 'CONFLICT'
    if row['material_ready_at'] is None or row['material_token_count'] is None:
        return 'UNKNOWN'
    if verdict is None:
        return 'UNKNOWN'
    return 'PASS' if verdict else 'FAIL'


def goodput_machine(rows, H=160, J=200, T=100, fault=''):
    if not 0 < T <= H <= J or len({row['name'] for row in rows}) != len(rows):
        raise ValueError('cutoff/identity')
    used_H = J if fault == 'evidence_moves_service_cutoff' else H
    entries = []
    out = dict(offered=0, confirmed_accepts=0, effective_admitted=0, rejected=0,
               preterminated=0, pending=0, normal=0, timeout=0, post_cancel=0, post_error=0, incomplete=0,
               quality_pass=0, quality_fail=0, quality_unknown=0, quality_conflict=0,
               known_good=0, unknown_good=0, definitely_not_good=0,
               cohort_output=0, known_good_tokens=0)
    for row in rows:
        if not 0 <= row['planned'] < T:
            continue
        out['offered'] += 1
        service = []
        if row['decision'] is not None:
            at, accepted = row['decision']
            service.append((at, 0, 0, 'decision', 'ADMIT', accepted))
        position = 0
        for i, (at, size) in enumerate(row['content']):
            service.append((at, 1, position, f'content-{i}', 'CONTENT', (position, size)))
            position += size
        if row['material_ready_at'] is not None:
            service.append((row['material_ready_at'], 2, 0, 'seal', 'SEAL',
                            (row['output_version'], row['material_token_count'])))
        if row['terminal'] is not None and row['terminal'][1] != 'REJECT':
            at, reason = row['terminal']
            service.append((at, 3, 0, 'terminal', 'TERMINAL', reason))
        state, seal, final, accepted_count = 'PENDING', None, None, 0
        seen, observed, cursor = {}, [], 0
        for at, _, _, identity, kind, payload in sorted(service):
            if at > used_H:
                continue
            if identity in seen:
                if seen[identity] != (at, kind, payload):
                    raise ValueError('service identity conflict')
                continue
            seen[identity] = (at, kind, payload)
            if kind == 'ADMIT':
                if type(payload) is not bool:
                    raise ValueError('admission type')
                accepted_count += payload
                if state.startswith('PRE_'):
                    continue
                if state != 'PENDING':
                    raise ValueError('second decision')
                state = 'ACTIVE' if payload else 'REJECTED'
            elif kind == 'CONTENT':
                start, size = payload
                if state != 'ACTIVE' or start != cursor or type(size) is not int or size <= 0:
                    raise ValueError('content/state/range')
                if row['mode'] == 'TOKEN' and size != 1:
                    raise ValueError('grouped TOKEN input')
                observed.append((at, size))
                cursor += size
            elif kind == 'SEAL':
                if seal is not None and seal != payload:
                    raise ValueError('mutable output identity')
                seal = payload
            elif kind == 'TERMINAL':
                if state == 'PENDING':
                    if payload == 'FINAL':
                        raise ValueError('FINAL without admission')
                    state = 'PRE_' + payload
                elif state == 'ACTIVE':
                    state = 'NORMAL' if payload == 'FINAL' else payload
                    final = at if payload == 'FINAL' else None
                else:
                    raise ValueError('business revival/duplicate terminal')
        if fault == 'infer_seal_from_last_client' and seal is None and observed:
            seal = (row['output_version'], cursor)
            row = dict(row, material_ready_at=observed[-1][0], material_token_count=cursor)
        last_client = observed[-1][0] if observed else None
        quality = certificate_machine(row, J, final, last_client, fault)
        if fault == 'quality_revives_business' and quality == 'PASS' and state != 'NORMAL':
            state, final = 'NORMAL', J
        if state == 'NORMAL' and seal is not None and seal != (row['output_version'], cursor):
            raise ValueError('normal output and sealed identity disagree')
        if state == 'NORMAL' and row['material_ready_at'] is not None and row['material_ready_at'] > final:
            raise ValueError('protocol FINAL before immutable output readiness')
        out['confirmed_accepts'] += accepted_count
        if state == 'PENDING':
            out['pending'] += 1
        elif state == 'REJECTED':
            out['rejected'] += 1
        elif state.startswith('PRE_'):
            out['preterminated'] += 1
        else:
            out['effective_admitted'] += 1
            if state == 'NORMAL':
                out['normal'] += 1
                out['cohort_output'] += cursor
            elif state == 'ACTIVE':
                out['incomplete'] += 1
            elif state == 'TIMEOUT':
                out['timeout'] += 1
            elif state == 'CANCEL':
                out['post_cancel'] += 1
            elif state == 'ERROR':
                out['post_error'] += 1
        if state != 'NORMAL':
            eligibility = 'FAIL'
            out['definitely_not_good'] += 1
        else:
            out['quality_' + quality.lower()] += 1
            exact = all(size == 1 for _, size in observed)
            gaps = [observed[i][0] - observed[i - 1][0] for i in range(1, len(observed))]
            tpot = F(sum(gaps), len(gaps)) if cursor > 1 and exact else None
            latency = False if not observed or observed[0][0] - row['sent'] > 35 else (
                True if cursor == 1 else None if tpot is None else tpot <= 15)
            if quality == 'CONFLICT':
                eligibility = 'CONFLICT'
            elif quality == 'FAIL' or latency is False:
                eligibility = 'FAIL'
                out['definitely_not_good'] += 1
            elif quality == 'UNKNOWN' or latency is None:
                eligibility = 'UNKNOWN'
                out['unknown_good'] += 1
            else:
                eligibility = 'PASS'
                out['known_good'] += 1
                out['known_good_tokens'] += cursor
        entries.append((row['name'], state, cursor, final, seal, quality, eligibility))
    integrity = out['quality_conflict'] == 0
    out['status'] = 'OK' if integrity else 'UNRESOLVED'
    out['goodput_lower'] = F(1000 * out['known_good'], used_H) if integrity else None
    out['goodput_upper'] = F(1000 * (out['known_good'] + out['unknown_good']), used_H) if integrity else None
    out['token_goodput_lower'] = F(1000 * out['known_good_tokens'], used_H) if integrity else None
    out['service_H'] = used_H
    out['per_request'] = tuple(entries)
    return out


def declaration_reference(rows, H=160, J=200, T=100):
    """Use endpoint predicates, identity payload sets and Boolean views, not replay."""
    facts = []
    for source in rows:
        if source['planned'] not in range(T):
            continue
        d = source['decision'] if source['decision'] and source['decision'][0] <= H else None
        z = source['terminal'] if source['terminal'] and source['terminal'][0] <= H else None
        before = bool(z and z[1] not in {'FINAL', 'REJECT'} and (d is None or z[0] < d[0]))
        if before:
            state = 'PRE_' + z[1]
        elif d is None:
            state = 'PENDING'
        elif not d[1]:
            state = 'REJECTED'
        elif z is None:
            state = 'ACTIVE'
        else:
            state = 'NORMAL' if z[1] == 'FINAL' else z[1]
        visible = [(at, n) for at, n in source['content'] if at <= H]
        count = sum(n for _, n in visible)
        material = source['material_ready_at']
        seal = (source['output_version'], source['material_token_count']) if material is not None and material <= H else None
        raw = ([] if source['quality'] is None else [
            ('primary', source['quality'][0], source['quality'][1], source['certificate_version'], RULE)])
        raw = raw + list(source['quality_events'])
        selected = [tuple(c) for c in raw if c[1] <= J and tuple(c[3:]) == (source['output_version'], RULE)]
        identity_sets = {}
        for identity, when, passed, _, _ in selected:
            identity_sets.setdefault(identity, set()).add((when, passed))
        verdict_set = {passed for _, _, passed, _, _ in selected}
        conflict = (any(len(values) > 1 for values in identity_sets.values()) or len(verdict_set) > 1 or
                    any(type(passed) is not bool or type(when) is not int or material is not None and when < material
                        for _, when, passed, _, _ in selected))
        quality = ('CONFLICT' if conflict else 'UNKNOWN' if not verdict_set or seal is None else
                   'PASS' if verdict_set == {True} else 'FAIL')
        normal = state == 'NORMAL'
        last = visible[-1][0] if visible else None
        exact = all(n == 1 for _, n in visible)
        avg_gap = F(last - visible[0][0], count - 1) if count > 1 and exact else None
        latency = False if count == 0 or visible[0][0] - source['sent'] > 35 else (
            True if count == 1 else None if avg_gap is None else avg_gap <= 15)
        if not normal:
            eligibility = 'FAIL'
        elif quality == 'CONFLICT':
            eligibility = 'CONFLICT'
        elif quality == 'FAIL' or latency is False:
            eligibility = 'FAIL'
        elif quality == 'UNKNOWN' or latency is None:
            eligibility = 'UNKNOWN'
        else:
            eligibility = 'PASS'
        facts.append(dict(name=source['name'], state=state, n=count, final=z[0] if normal else None,
                          seal=seal, q=quality, c=eligibility, accepted=bool(d and d[1])))
    names_by_state = {name: {r['name'] for r in facts if r['state'] == name}
                      for name in ['PENDING', 'REJECTED', 'NORMAL', 'ACTIVE', 'TIMEOUT', 'CANCEL', 'ERROR']}
    pre = {r['name'] for r in facts if r['state'].startswith('PRE_')}
    admitted = {r['name'] for r in facts if r['state'] not in {'PENDING', 'REJECTED'} and r['name'] not in pre}
    normal = [r for r in facts if r['state'] == 'NORMAL']
    good = [r for r in facts if r['c'] == 'PASS']
    unknown = [r for r in facts if r['c'] == 'UNKNOWN']
    conflicts = [r for r in normal if r['q'] == 'CONFLICT']
    out = dict(offered=len(facts), confirmed_accepts=sum(r['accepted'] for r in facts),
               effective_admitted=len(admitted), rejected=len(names_by_state['REJECTED']),
               preterminated=len(pre), pending=len(names_by_state['PENDING']), normal=len(normal),
               timeout=len(names_by_state['TIMEOUT']), post_cancel=len(names_by_state['CANCEL']),
               post_error=len(names_by_state['ERROR']), incomplete=len(names_by_state['ACTIVE']),
               quality_pass=sum(r['q'] == 'PASS' for r in normal),
               quality_fail=sum(r['q'] == 'FAIL' for r in normal),
               quality_unknown=sum(r['q'] == 'UNKNOWN' for r in normal), quality_conflict=len(conflicts),
               known_good=len(good), unknown_good=len(unknown),
               definitely_not_good=sum(r['c'] == 'FAIL' for r in facts),
               cohort_output=sum(r['n'] for r in normal), known_good_tokens=sum(r['n'] for r in good),
               status='UNRESOLVED' if conflicts else 'OK', service_H=H)
    out['goodput_lower'] = None if conflicts else F(1000 * len(good), H)
    out['goodput_upper'] = None if conflicts else F(1000 * (len(good) + len(unknown)), H)
    out['token_goodput_lower'] = None if conflicts else F(1000 * sum(r['n'] for r in good), H)
    out['per_request'] = tuple((r['name'], r['state'], r['n'], r['final'], r['seal'], r['q'], r['c']) for r in facts)
    return out


def changed(rows, name, **updates):
    return [dict(row, **updates) if row['name'] == name else deepcopy(row) for row in rows]


def summary(report):
    keys = ['offered', 'confirmed_accepts', 'effective_admitted', 'normal', 'known_good',
            'unknown_good', 'cohort_output', 'goodput_lower', 'goodput_upper', 'token_goodput_lower']
    return {key: str(report[key]) if isinstance(report[key], F) else report[key] for key in keys}


def main():
    fingerprint = hashlib.sha256(json.dumps(COMMON_FIXTURE, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    same(fingerprint, COMMON_FIXTURE_SHA256, 'common fixture fingerprint')
    base = goodput_machine(COMMON_FIXTURE)
    same(base, declaration_reference(COMMON_FIXTURE), 'base replay/reference')
    same(summary(base), dict(offered=8, confirmed_accepts=7, effective_admitted=6, normal=4,
                            known_good=3, unknown_good=1, cohort_output=10,
                            goodput_lower='75/4', goodput_upper='25', token_goodput_lower='175/4'), 'shared hand census')
    cases = [('base', COMMON_FIXTURE, 160, 200), ('online-evidence-cutoff', COMMON_FIXTURE, 160, 160)]
    early = changed(COMMON_FIXTURE, 'A', quality=[35, True])
    producer_early = changed(COMMON_FIXTURE, 'A', material_ready_at=25, quality=[28, True])
    conflict = changed(COMMON_FIXTURE, 'A', quality=[35, False],
                       quality_events=[['second', 40, True, 'v1', RULE]])
    identity_conflict = changed(COMMON_FIXTURE, 'A',
                                quality_events=[['primary', 181, True, 'v1', RULE]])
    duplicate = changed(COMMON_FIXTURE, 'A',
                        quality_events=[['primary', 180, True, 'v1', RULE]])
    missing_material = changed(COMMON_FIXTURE, 'A', material_ready_at=None, material_token_count=None)
    future_final = changed(COMMON_FIXTURE, 'F', quality=[120, True])
    cancelled_complete_content = changed(COMMON_FIXTURE, 'A', terminal=[45, 'CANCEL'])
    cases.extend([('30/35/45', early, 160, 160), ('25/28/30/45', producer_early, 160, 160),
                  ('FAIL then PASS', conflict, 160, 200), ('identity conflict', identity_conflict, 160, 200),
                  ('duplicate same proof', duplicate, 160, 200), ('missing immutable output', missing_material, 160, 200),
                  ('future FINAL stays outside H', future_final, 160, 200),
                  ('certificate never revives cancelled business', cancelled_complete_content, 160, 200),
                  ('J does not yet see quality', COMMON_FIXTURE, 160, 179),
                  ('J exactly includes quality', COMMON_FIXTURE, 160, 180)])
    checks = 2
    for name, rows, H, J in cases:
        actual, reference = goodput_machine(rows, H, J), declaration_reference(rows, H, J)
        same(actual, reference, name)
        same(actual['offered'], actual['pending'] + actual['rejected'] + actual['preterminated'] + actual['effective_admitted'], name + ' offered conservation')
        same(actual['normal'], sum(actual['quality_' + s] for s in ['pass', 'fail', 'unknown', 'conflict']), name + ' quality conservation')
        same(actual['effective_admitted'], actual['normal'] + actual['timeout'] + actual['post_cancel'] + actual['post_error'] + actual['incomplete'], name + ' admission conservation')
        checks += 4
    same(goodput_machine(early, J=160)['known_good'], 3, 'early proof is eligible after business FINAL')
    same(goodput_machine(producer_early, J=160)['known_good'], 3, 'producer seal is not client last-content time')
    same(goodput_machine(conflict)['status'], 'UNRESOLVED', 'fixed rule contradiction')
    same(goodput_machine(conflict)['goodput_lower'], None, 'conflict cannot produce certified lower report')
    same(goodput_machine(future_final)['normal'], 4, 'J does not add future business FINAL')
    same(goodput_machine(duplicate), base, 'idempotence')
    checks += 6
    faults = [
        ('quality_requires_final', early, 160, 160),
        ('quality_requires_client_last', producer_early, 160, 160),
        ('last_verdict_wins', conflict, 160, 200),
        ('include_future_certificate', COMMON_FIXTURE, 160, 160),
        ('evidence_moves_service_cutoff', future_final, 160, 200),
        ('quality_revives_business', future_final, 160, 200),
        ('infer_seal_from_last_client', missing_material, 160, 200),
    ]
    detected = []
    for fault, rows, H, J in faults:
        try:
            same(goodput_machine(rows, H, J, fault=fault), declaration_reference(rows, H, J), fault)
        except CheckError:
            detected.append(fault)
        else:
            raise CheckError('undetected mutant: ' + fault)
    checks += len(detected)
    same(goodput_machine([]), declaration_reference([]), 'empty cohort')
    checks += 1
    print(json.dumps({'result': 'PASS', 'checks': checks, 'mutants_rejected': detected,
                      'common_fixture_sha256': fingerprint, 'common_fixture_summary': summary(base),
                      'scope': 'Finite synthetic independent service-state and declarative certificate models; no real evaluator, service/GPU behavior, population SLO coverage or Agent-effectiveness claim.'}, sort_keys=True))


if __name__ == '__main__':
    main()
