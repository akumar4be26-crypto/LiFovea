import unittest

from avr25d.server.app import system_telemetry


class TestTelemetry(unittest.TestCase):
    def test_returns_expected_keys_and_ranges(self):
        stats = system_telemetry()

        for key in (
            "cpu_percent",
            "ram_percent",
            "gpu",
            "gpu_available",
            "temperature_c",
            "power_watts",
            "source",
            "device",
        ):
            self.assertIn(key, stats)

        self.assertTrue(0 <= stats["cpu_percent"] <= 100)
        self.assertTrue(0 <= stats["ram_percent"] <= 100)
        self.assertTrue(0 <= stats["power_watts"] <= 500)
        self.assertIn(stats["gpu"], {"CPU", "Metal", "CUDA", "Unknown"})
        self.assertIn(stats["source"], {"direct", "estimated"})


if __name__ == "__main__":
    unittest.main()
