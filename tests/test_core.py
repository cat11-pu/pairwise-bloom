"""Behaviour tests for the bloom filter core."""

import unittest

from bloom.core import BitMap
from bloom.core import BloomFilter
from bloom.core import CountingBloomFilter
from bloom.core import DedupWindow
from bloom.core import GrowingBloomFilter


class TestBitMap(unittest.TestCase):
    def test_bits_are_set_read_and_cleared_in_place(self):
        bits = BitMap(10)
        bits.set_bit(0)
        bits.set_bit(1)
        bits.set_bit(9)
        self.assertEqual(bits.to_bytes(), bytes([0x03, 0x02]))
        self.assertEqual([bits.get_bit(index) for index in (0, 1, 2, 9)], [1, 1, 0, 1])
        bits.clear_bit(1)
        self.assertEqual(bits.get_bit(1), 0)
        self.assertEqual(bits.get_bit(0), 1)
        self.assertEqual(bits.get_bit(9), 1)
        bits.clear_bit(9)
        self.assertEqual(bits.get_bit(9), 0)
        self.assertEqual(bits.count(), 1)

    def test_the_last_partial_byte_is_counted(self):
        bits = BitMap(13)
        self.assertEqual(bits.count(), 0)
        for position in range(13):
            bits.set_bit(position)
        self.assertEqual(bits.count(), 13)
        bits.clear_bit(12)
        self.assertEqual(bits.count(), 12)

    def test_positions_outside_the_bitmap_are_rejected(self):
        bits = BitMap(13)
        with self.assertRaises(ValueError):
            bits.set_bit(13)
        with self.assertRaises(ValueError):
            bits.get_bit(13)
        with self.assertRaises(ValueError):
            bits.clear_bit(13)
        with self.assertRaises(ValueError):
            bits.set_bit(-1)
        with self.assertRaises(TypeError):
            bits.set_bit("3")
        with self.assertRaises(ValueError):
            BitMap(0)


class TestBloomFilter(unittest.TestCase):
    def test_every_recorded_key_is_found_again(self):
        bloom = BloomFilter(capacity=64, error_rate=0.01)
        keys = ["member-%03d" % index for index in range(64)]
        for key in keys:
            bloom.add(key)
        for key in keys:
            self.assertTrue(bloom.contains(key), key)
        self.assertEqual(bloom.count, 64)
        self.assertGreater(bloom.m, 64)
        self.assertGreaterEqual(bloom.k, 1)

    def test_keys_that_were_never_recorded_are_not_reported(self):
        bloom = BloomFilter(capacity=32, error_rate=0.01)
        for index in range(32):
            bloom.add("member-%03d" % index)
        rate = bloom.estimated_fp_rate()
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.5)
        self.assertFalse(bloom.contains("outsider-000"))
        self.assertFalse(bloom.contains("outsider-001"))

    def test_positions_and_set_bits_stay_consistent(self):
        bloom = BloomFilter(capacity=8, error_rate=0.01)
        covered = set()
        for index in range(12):
            key = "key-%02d" % index
            bloom.add(key)
            covered.update(bloom.positions(key))
        self.assertEqual(len(bloom.positions("key-00")), bloom.k)
        self.assertEqual(bloom.bits_set(), len(covered))
        self.assertAlmostEqual(bloom.fill_ratio(), bloom.bits_set() / bloom.m, places=12)
        self.assertAlmostEqual(
            bloom.estimated_fp_rate(), bloom.fill_ratio() ** bloom.k, places=12
        )


class TestCountingBloomFilter(unittest.TestCase):
    def test_removing_an_unknown_key_is_reported(self):
        counts = CountingBloomFilter(capacity=8, error_rate=0.01)
        counts.add("resident-a")
        counts.add("resident-b")
        self.assertFalse(counts.contains("visitor"))
        self.assertFalse(counts.remove("visitor"))

    def test_counters_never_drop_below_zero(self):
        counts = CountingBloomFilter(capacity=8, error_rate=0.01)
        for key in ("alpha", "bravo", "charlie"):
            counts.add(key)
        self.assertTrue(counts.remove("alpha"))
        positions = counts.positions("visitor")
        counts.remove("visitor")
        self.assertEqual([counts.counter_at(p) for p in positions], [0] * len(positions))
        self.assertGreaterEqual(min(counts.counter_at(p) for p in range(counts.m)), 0)
        for key in ("bravo", "charlie"):
            self.assertTrue(counts.contains(key))


class TestGrowingBloomFilter(unittest.TestCase):
    def test_every_generation_of_keys_survives_the_rebuilds(self):
        growing = GrowingBloomFilter(capacity=8, error_rate=0.01, growth=2)
        keys = ["gen-%02d" % index for index in range(20)]
        for key in keys:
            growing.add(key)
        self.assertGreaterEqual(growing.rebuilds, 2)
        self.assertEqual(growing.capacity, 32)
        self.assertGreater(growing.m, BloomFilter(capacity=8, error_rate=0.01).m)
        for key in keys:
            self.assertTrue(growing.contains(key), key)


class TestDedupWindow(unittest.TestCase):
    def test_keys_leave_the_window_and_the_reused_slot_is_emptied(self):
        window = DedupWindow(window=8, slot_size=4, error_rate=0.0001)
        first = ["alpha", "bravo", "charlie", "delta"]
        for key in first:
            self.assertFalse(window.seen(key))
        for key in first:
            self.assertTrue(window.seen(key))
        second = ["echo", "foxtrot", "golf", "hotel"]
        for key in second:
            self.assertFalse(window.seen(key))
        third = ["india", "juliet", "kilo", "lima"]
        for key in third:
            self.assertFalse(window.seen(key))
        self.assertLess(max(window.slot_loads()), 60)
        self.assertTrue(window.seen("echo"))
        self.assertFalse(window.seen("alpha"))
