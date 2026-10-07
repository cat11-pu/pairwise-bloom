"""Bloom filter core: bitmaps, multi hash filters, counting and growing variants.

Keys are digested with the local 64 bit helpers instead of the built in
hash(), so a filter built from the same keys always ends up with exactly the
same bits and counters.  Everything is in memory and deterministic.
"""

import math

MASK64 = (1 << 64) - 1
FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
SPLITMIX_SALT = 0x9E3779B97F4A7C15


class BloomError(Exception):
    """Raised when a filter cannot carry out the requested operation."""


def _popcount(value):
    """Number of set bits in a single byte."""
    return bin(value).count("1")


def _as_bytes(key):
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise TypeError("key must be str or bytes")


def fnv1a64(key):
    """64 bit FNV-1a digest of key."""
    digest = FNV_OFFSET_BASIS
    for byte in _as_bytes(key):
        digest ^= byte
        digest = (digest * FNV_PRIME) & MASK64
    return digest


def mix64(value):
    """SplitMix64 finalizer: turns a counter into well mixed bits."""
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return (value ^ (value >> 31)) & MASK64


def _steps(key, nbits, k):
    """The positions one key occupies: first + step * second, modulo nbits."""
    first = fnv1a64(key)
    second = mix64(first ^ SPLITMIX_SALT) | 1
    return [(first + step * second) % nbits for step in range(k)]


def _check_capacity(capacity):
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an int")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return capacity


def _check_error_rate(error_rate):
    if isinstance(error_rate, bool) or not isinstance(error_rate, (int, float)):
        raise TypeError("error_rate must be a number")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must be between 0 and 1")
    return float(error_rate)


def _params(capacity, error_rate):
    """Bitmap width m and hash count k for capacity keys at error_rate."""
    m = int(math.ceil(-capacity * math.log(error_rate) / (math.log(2.0) ** 2)))
    k = int(round(m / capacity * math.log(2.0)))
    return max(8, m), max(1, k)


class BitMap:
    """A fixed length bit vector.

    Bit 0 is the low bit of byte 0, bit 8 is the low bit of byte 1, and the
    last byte is only partly used when nbits is not a multiple of eight.
    """

    def __init__(self, nbits):
        if isinstance(nbits, bool) or not isinstance(nbits, int):
            raise TypeError("nbits must be an int")
        if nbits <= 0:
            raise ValueError("nbits must be positive")
        self.nbits = nbits
        self._bytes = bytearray(self.nbytes())

    def nbytes(self):
        """How many bytes back the bitmap."""
        return (self.nbits + 7) // 8

    def _check_position(self, position):
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("position must be an int")
        if position < 0 or position >= self.nbits:
            raise ValueError("position %r is outside the bitmap" % (position,))
        return position

    def set_bit(self, position):
        """Turn the bit at position on."""
        position = self._check_position(position)
        self._bytes[position >> 3] |= 1 << (position & 7)

    def get_bit(self, position):
        """The value of the bit at position, 0 or 1."""
        position = self._check_position(position)
        return (self._bytes[position >> 3] >> (position & 7)) & 1

    def clear_bit(self, position):
        """Turn the bit at position off and leave its byte mates alone."""
        position = self._check_position(position)
        index = position >> 3
        self._bytes[index] &= 0xFF ^ (1 << (position & 7))

    def clear(self):
        """Drop every bit, keeping the same buffer for reuse."""
        for position in range(self.nbits):
            self.clear_bit(position)

    def count(self):
        """How many bits are currently set."""
        total = sum(_popcount(value) for value in self._bytes)
        remainder = self.nbits % 8
        if remainder:
            total -= _popcount(self._bytes[-1] >> remainder << remainder)
        return total

    def to_bytes(self):
        """The raw backing bytes."""
        return bytes(self._bytes)


class BloomFilter:
    """A classic bloom filter: m bits wide, k hashes per key, no removals."""

    def __init__(self, capacity, error_rate=0.01):
        self.capacity = _check_capacity(capacity)
        self.error_rate = _check_error_rate(error_rate)
        self.m, self.k = _params(self.capacity, self.error_rate)
        self.count = 0
        self._bits = BitMap(self.m)

    def positions(self, key):
        """The bit positions key occupies."""
        return _steps(key, self.m, self.k)

    def add(self, key):
        """Record key in the bitmap."""
        for position in self.positions(key):
            self._bits.set_bit(position)
        self.count += 1

    def contains(self, key):
        """Whether key is very likely to have been recorded."""
        return all(self._bits.get_bit(position) for position in self.positions(key))

    def bits_set(self):
        """How many bits of the bitmap are on."""
        return self._bits.count()

    def fill_ratio(self):
        """The fraction of the bitmap that is on."""
        return self.bits_set() / self.m

    def estimated_fp_rate(self):
        """False positive estimate from the current fill: fill ** k."""
        return self.fill_ratio() ** self.k

    def reset(self):
        """Forget every key, reusing the same bit buffer."""
        self._bits.clear()
        self.count = 0


class CountingBloomFilter:
    """A bloom filter whose positions carry counters, so keys can be dropped."""

    def __init__(self, capacity, error_rate=0.01, limit=255):
        self.capacity = _check_capacity(capacity)
        self.error_rate = _check_error_rate(error_rate)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an int")
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        self.m, self.k = _params(self.capacity, self.error_rate)
        self.count = 0
        self._counters = [0] * self.m

    def positions(self, key):
        """The counter slots key occupies."""
        return _steps(key, self.m, self.k)

    def add(self, key):
        """Record one occurrence of key."""
        for position in self.positions(key):
            if self._counters[position] < self.limit:
                self._counters[position] += 1
        self.count += 1

    def remove(self, key):
        """Drop one occurrence of key and report whether it was there."""
        if not self.contains(key):
            return False
        for position in self.positions(key):
            self._counters[position] -= 1
        self.count -= 1
        return True

    def contains(self, key):
        """Whether every slot of key still holds a positive counter."""
        return all(self._counters[position] > 0 for position in self.positions(key))

    def counter_at(self, position):
        """The counter stored at position."""
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError("position must be an int")
        if position < 0 or position >= self.m:
            raise ValueError("position %r is outside the counters" % (position,))
        return self._counters[position]

    def fill_ratio(self):
        """The fraction of slots that hold a positive counter."""
        return sum(1 for value in self._counters if value > 0) / self.m

    def estimated_fp_rate(self):
        """False positive estimate from the current fill: fill ** k."""
        return self.fill_ratio() ** self.k


class GrowingBloomFilter:
    """A bloom filter that rebuilds wider once it is full.

    Every key handed to add() is remembered so it can be replayed into the
    next, wider bitmap.
    """

    def __init__(self, capacity, error_rate=0.01, growth=2):
        self.capacity = _check_capacity(capacity)
        self.error_rate = _check_error_rate(error_rate)
        if isinstance(growth, bool) or not isinstance(growth, int):
            raise TypeError("growth must be an int")
        if growth < 2:
            raise ValueError("growth must be at least 2")
        self.growth = growth
        self.rebuilds = 0
        self._keys = []
        self._filter = BloomFilter(self.capacity, self.error_rate)

    @property
    def m(self):
        return self._filter.m

    @property
    def k(self):
        return self._filter.k

    def add(self, key):
        """Record key, rebuilding the bitmap when it is full."""
        if self._filter.count >= self.capacity:
            self._grow()
        self._keys.append(key)
        self._filter.add(key)

    def contains(self, key):
        """Whether key is very likely to have been recorded."""
        return self._filter.contains(key)

    def bits_set(self):
        return self._filter.bits_set()

    def _grow(self):
        capacity = self.capacity * self.growth
        grown = BloomFilter(capacity, self.error_rate)
        for key in self._keys:
            grown.add(key)
        self.capacity = capacity
        self._filter = grown
        self.rebuilds += 1


class DedupWindow:
    """Duplicate window: the keys of the last few slotfuls are remembered.

    The window is a ring of bitmaps.  A slot takes slot_size keys and is then
    handed to the next batch, so the oldest keys leave the window.
    """

    def __init__(self, window, slot_size=1, error_rate=0.001):
        self.window = _check_capacity(window)
        self.slot_size = _check_capacity(slot_size)
        self.error_rate = _check_error_rate(error_rate)
        if self.window % self.slot_size:
            raise ValueError("window must be a multiple of slot_size")
        self._slots = [
            BloomFilter(self.slot_size, self.error_rate)
            for _ in range(self.window // self.slot_size)
        ]
        self._cursor = 0
        self._used = 0

    def seen(self, key):
        """Record key; True when key was already inside the window."""
        for slot in self._slots:
            if slot.contains(key):
                return True
        self._record(key)
        return False

    def slot_loads(self):
        """How many bits each slot currently has on."""
        return tuple(slot.bits_set() for slot in self._slots)

    def _record(self, key):
        if self._used >= self.slot_size:
            self._advance()
        self._slots[self._cursor].add(key)
        self._used += 1

    def _advance(self):
        self._cursor = (self._cursor + 1) % len(self._slots)
        self._slots[self._cursor].reset()
        self._used = 0
