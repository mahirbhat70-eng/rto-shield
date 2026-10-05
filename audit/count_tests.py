import pytest
import collections

class CollectorPlugin:
    def __init__(self):
        self.counts = collections.defaultdict(int)

    def pytest_collection_modifyitems(self, items):
        for item in items:
            self.counts[item.fspath.basename] += 1

collector = CollectorPlugin()
pytest.main(["--collect-only", "-q"], plugins=[collector])
print("\n--- TEST COUNTS PER FILE ---")
for f, n in sorted(collector.counts.items()):
    print(f"{f}: {n}")
print(f"TOTAL: {sum(collector.counts.values())}")
