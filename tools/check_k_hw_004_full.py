#!/usr/bin/env python3
"""K-HW-004: deterministic read-only scalar/reference and handoff-model checks.
No target kernel, GPU memory semantics, timing, or model quality is tested.
"""

from fractions import Fraction

INT32_MAX = (1 << 31) - 1


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except kind:
        return
    raise AssertionError('expected ' + kind.__name__ + ' from ' + function.__name__)


def require_int4(bits):
    if type(bits) is not int or bits != 4:
        raise ValueError('SYM4-ROW-LOWFIRST-v1 accepts exactly bits=4')


def pack_signed(codes, bits=4):
    require_int4(bits)
    codes = list(codes)
    if any(type(c) is not int or not -8 <= c <= 7 for c in codes):
        raise ValueError('invalid signed INT4 code')
    out = bytearray()
    for i in range(0, len(codes), 2):
        low = codes[i] & 15
        high = codes[i + 1] & 15 if i + 1 < len(codes) else 0
        out.append(low | (high << 4))
    return bytes(out)


def unpack_signed(payload, count, bits=4):
    require_int4(bits)
    if type(count) is not int or count < 0:
        raise ValueError('invalid valid-element count')
    if not isinstance(payload, bytes) or len(payload) != (count + 1) // 2:
        raise ValueError('payload length mismatch')
    if count % 2 and payload[-1] >> 4:
        raise ValueError('nonzero unused high nibble')
    result = []
    for i in range(count):
        nibble = (payload[i // 2] >> (4 * (i % 2))) & 15
        result.append(nibble - 16 if nibble >= 8 else nibble)
    return result


def unsigned_unpack_fault(payload, count):
    return [(payload[i // 2] >> (4 * (i % 2))) & 15 for i in range(count)]


def old_fixed_shift_fault(pair, bits):
    mask = (1 << bits) - 1
    return (pair[0] & mask) | ((pair[1] & mask) << 4)


def nearest_half_up(value):
    value = Fraction(value)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def quantize_row(values, group, save_scale=lambda value: value):
    if type(group) is not int or group <= 0:
        raise ValueError('invalid group')
    values = tuple(Fraction(value) for value in values)
    params = []
    lengths = []
    for start in range(0, len(values), group):
        part = values[start:start + group]
        maximum = max(abs(value) for value in part)
        candidate = maximum / 7 if maximum else Fraction(1)
        stored = Fraction(save_scale(candidate))
        if stored <= 0:
            raise ValueError('stored scale is not positive')
        params.append(stored)
        lengths.append(len(part))
    frozen = tuple(params)
    codes = []
    for i, value in enumerate(values):
        code = nearest_half_up(value / frozen[i // group])
        codes.append(max(-8, min(7, code)))
    return pack_signed(codes), frozen, tuple(lengths), tuple(codes)


def nearest_grid_reference(value, scale):
    # Independent enumeration: nearest reconstructed value, larger code on ties.
    return min(range(-8, 8), key=lambda code: (abs(scale * code - value), -code))


def grouped_dot(codes, activations, scales, group):
    total = Fraction(0)
    for g, scale in enumerate(scales):
        partial = sum(codes[i] * activations[i]
                      for i in range(g * group, min((g + 1) * group, len(codes))))
        total += scale * partial
    return total


def dequantized_dot_reference(codes, activations, scales, group):
    return sum((scales[i // group] * code) * activations[i]
               for i, code in enumerate(codes))


def packed_weight_bytes(rows, cols, group, meta_bytes):
    code_bytes = rows * ((cols + 1) // 2)
    metadata = rows * ((cols + group - 1) // group) * meta_bytes
    return code_bytes, metadata


def int8_abs_bound(k, initial=0, bias=0):
    return k * 128 * 128 + abs(initial) + abs(bias)


def wmma_address_legality(lda, ldb, ldc, base_a, base_b, base_c):
    dimensions = (lda, ldb, ldc)
    bases = (base_a, base_b, base_c)
    return (all(type(v) is int and v > 0 for v in dimensions)
            and all(type(v) is int and v >= 0 for v in bases)
            and lda % 8 == 0 and ldb % 8 == 0 and ldc % 4 == 0
            and all(v % 32 == 0 for v in bases))


class SharedModel:
    """Abstract ordinary shared writes; publication is an explicit model action."""

    def __init__(self):
        self.visible = [0] * 256
        self.pending = {}

    def write(self, index, value):
        require(index not in self.pending, 'duplicate writer for a shared position')
        self.pending[index] = value

    def publish(self):
        require(set(self.pending) == set(range(256)), 'tile was not completely written')
        for index, value in self.pending.items():
            self.visible[index] = value
        self.pending.clear()

    def load(self):
        return self.visible[:]


def dense_reference(a, b, n):
    # Never reads staged buffers, masks, or model fragments.
    return [[sum(a[i][k] * b[k][j] for k in range(len(a[i])))
             for j in range(n)] for i in range(len(a))]


def staged_tile_model(m, n, k_size, fault=''):
    require(0 < m <= 16 and 0 < n <= 16 and k_size >= 0, 'model domain')
    a = [[(3 * i + k) % 5 - 2 for k in range(k_size)] for i in range(m)]
    b = [[(k + 2 * j) % 7 - 3 for j in range(n)] for k in range(k_size)]
    as_model, bs_model, cs_model = SharedModel(), SharedModel(), SharedModel()
    accumulator = [0] * 256
    reusable = True
    for k0 in range(0, k_size, 16):
        require(reusable, 'reuse before all readers')
        for lane in range(32):
            for u in range(8):
                t = lane + 32 * u
                i, k = divmod(t, 16)
                av = a[i][k0 + k] if i < m and k0 + k < k_size else 0
                as_model.write(t, av)
                k, j = divmod(t, 16)
                bv = b[k0 + k][j] if k0 + k < k_size and j < n else 0
                bs_model.write(t, bv)
        if fault != 'load_before_publish':
            as_model.publish()
            bs_model.publish()
        af = as_model.load()
        bf = bs_model.load()
        reusable = False
        if fault == 'load_before_publish':
            # A barrier after loading cannot repair values already copied to af/bf.
            as_model.publish()
            bs_model.publish()
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    accumulator[16 * i + j] += af[16 * i + k] * bf[16 * k + j]
        if fault != 'omit_read_done':
            reusable = True
    for t in range(256):
        cs_model.write(t, accumulator[t])
    if fault != 'omit_output_publish':
        cs_model.publish()
    stored = cs_model.load()
    result = [[stored[16 * i + j] for j in range(n)] for i in range(m)]
    return result, dense_reference(a, b, n)


def main():
    print('scope: Python scalar/rational arithmetic and abstract tile publication model; no target CPU/GPU kernels')

    seen = set()
    for low in range(-8, 8):
        for high in range(-8, 8):
            payload = pack_signed([low, high])
            # Reference uses arithmetic representation, no packer bit operations.
            expected = (low + 16 if low < 0 else low) + 16 * (high + 16 if high < 0 else high)
            require(payload == bytes([expected]), 'independent byte reference mismatch')
            require(unpack_signed(payload, 2) == [low, high], 'pair round trip')
            seen.add(payload)
    require(len(seen) == 256, 'INT4 pair encoding collision')
    for codes in ([], [-8], [-1], [7], [-8, -7, -1, 0, 1, 7, 3]):
        payload = pack_signed(codes)
        require(unpack_signed(payload, len(codes)) == codes, 'edge round trip')
        if len(codes) % 2:
            require(payload[-1] < 16, 'odd-tail high nibble is not zero')
    require(pack_signed([-1, 1]) == bytes([0x1F]), 'known signed byte')
    require(unsigned_unpack_fault(bytes([0x1F]), 2) != [-1, 1], 'unsigned fault missed')
    for invalid in ([-9], [8], [1.0], [True]):
        expect_error(ValueError, pack_signed, invalid)
    for bits in (2, 3, 5, 8, 16):
        expect_error(ValueError, pack_signed, [0, 1], bits=bits)
        expect_error(ValueError, unpack_signed, bytes([0x10]), 2, bits=bits)
    for pair in ([16, 0], [0, 1]):
        expect_error(ValueError, pack_signed, pair, bits=8)
    require(old_fixed_shift_fault([16, 0], 8) == old_fixed_shift_fault([0, 1], 8) == 16,
            'original non-INT4 collision witness changed')
    expect_error(ValueError, unpack_signed, bytes([0xF1]), 1)
    expect_error(ValueError, unpack_signed, b'', 1)
    expect_error(ValueError, unpack_signed, bytes([0]), 0)
    expect_error(ValueError, unpack_signed, b'', -1)
    rows = [[-1, 1, 7], [-8, 0, -1], [3, 2, 1]]
    packed_rows = [pack_signed(row) for row in rows]
    require(len(b''.join(packed_rows)) == 6, 'per-row odd tail capacity')
    for row, payload in zip(rows, packed_rows):
        require(unpack_signed(payload, 3) == row, 'row identity lost')
    print('PASS 256 distinct INT4 pair bytes; empty/odd/row tails, malformed inputs, unsigned fault, and non-4-bit rejection')

    values = [Fraction(0), Fraction(0), Fraction(0), Fraction(1, 2),
              Fraction(3, 2), Fraction(-3, 2), Fraction(7, 4)]
    # Synthetic exact scale-storage grid, not a claim to emulate IEEE FP16.
    save_eighth = lambda value: Fraction(nearest_half_up(value * 8), 8)
    payload, params, lengths, codes = quantize_row(values, 3, save_eighth)
    require(params == (Fraction(1), Fraction(1, 4), Fraction(1, 4)), 'stored scale reference')
    require(lengths == (3, 3, 1), 'short group lengths')
    expected_codes = tuple(nearest_grid_reference(value, params[i // 3])
                           for i, value in enumerate(values))
    require(codes == expected_codes == (0, 0, 0, 2, 6, -6, 7), 'enumerated grid encoding')
    decoded = unpack_signed(payload, len(values))
    require(tuple(decoded) == codes, 'quantized payload codes')
    reconstructed = [params[i // 3] * code for i, code in enumerate(decoded)]
    require(reconstructed == [params[i // 3] * code for i, code in enumerate(expected_codes)],
            'frozen parameter reconstruction')
    altered = [2 * params[i // 3] * code for i, code in enumerate(decoded)]
    require(altered != reconstructed, 'changed-export-scale fault missed')
    for value in (Fraction(-1, 2), Fraction(1, 2), Fraction(-3, 2), Fraction(3, 2)):
        q = nearest_half_up(value)
        require(q == nearest_grid_reference(value, Fraction(1)), 'half-integer tie reference')
    require(quantize_row([2, 2], 2)[3] == (7, 7), 'nonzero constant group')
    require(quantize_row([], 2)[:3] == (b'', (), ()), 'empty quantization')
    expect_error(ValueError, quantize_row, [Fraction(1, 1000)], 1, save_eighth)
    print('PASS rational encoder versus enumerated grid; zero/constant/short groups, saved scales, ties, zero-scale rejection and export-scale fault')

    codes = [1, 1, 1, 1, -2]
    activations = [1, 1, 1, 1, 3]
    scales = [Fraction(1, 2), Fraction(2), Fraction(1, 4)]
    ref = dequantized_dot_reference(codes, activations, scales, 2)
    require(grouped_dot(codes, activations, scales, 2) == ref == Fraction(7, 2), 'grouped dot')
    simple_ref = dequantized_dot_reference([1] * 4, [1] * 4, scales[:2], 2)
    bad_epilogue = sum([1] * 4) * scales[1]
    require(simple_ref == 5 and bad_epilogue == 8, 'single epilogue fault missed')
    code_bytes, metadata = packed_weight_bytes(4096, 4096, 128, 4)
    require((code_bytes, metadata) == (8388608, 524288), 'capacity reference')
    require(code_bytes + metadata == 8912896, 'total capacity')
    require(Fraction(8 * (code_bytes + metadata), 4096 * 4096) == Fraction(17, 4), 'effective bits')
    require(packed_weight_bytes(3, 3, 2, 0)[0] == 6, 'row padding capacity')
    require((3 * 3 * 4 + 7) // 8 == 5, 'continuous-stream undercount witness')
    require(int8_abs_bound(4096) == 67108864 <= INT32_MAX, 'short INT8 bound')
    require(int8_abs_bound(131072) > INT32_MAX, 'long INT8 bound not rejected')
    require(131072 * 127 * 127 <= INT32_MAX, '127-squared faulty bound witness')
    require(int8_abs_bound(0, -3, 5) == 8, 'initial and bias contribution')
    print('PASS grouped scales, row-padded capacity=8912896/effective_bits=4.25, and INT8 intermediate bounds; faulty epilogue/byte count/127^2 controls detected')

    require(wmma_address_legality(16, 16, 16, 0, 512, 1024), 'legal address model')
    require(not wmma_address_legality(15, 16, 16, 0, 512, 1024), 'bad lda accepted')
    require(not wmma_address_legality(16, 16, 16, 0, 512, 1040), 'misaligned C_s accepted')
    require(not wmma_address_legality(16, 16, 14, 0, 512, 1024), 'bad ldc accepted')
    positions = [lane + 32 * u for lane in range(32) for u in range(8)]
    require(sorted(positions) == list(range(256)), 'cooperative tile ownership')
    for m, n, k in ((3, 5, 0), (3, 5, 1), (3, 5, 16), (3, 5, 17), (16, 16, 32)):
        actual, reference = staged_tile_model(m, n, k)
        require(actual == reference, 'staged tile versus independent dense reference')
    for fault in ('load_before_publish', 'omit_output_publish'):
        actual, reference = staged_tile_model(3, 5, 17, fault)
        require(actual != reference, fault + ' negative control missed')
    expect_error(AssertionError, staged_tile_model, 3, 5, 17, 'omit_read_done')
    print('PASS tile mapping and 5 dense-reference cases; load-before-publication, missing output publication and early-reuse faults detected in abstract model')
    print('LIMIT: no Python test result implies GPU visibility, real WMMA execution, FP16/FP32 rounding coverage, dispatch, transactions, performance or model quality')


if __name__ == '__main__':
    main()
