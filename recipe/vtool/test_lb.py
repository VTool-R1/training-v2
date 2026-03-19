from __future__ import annotations

import asyncio
import unittest

from recipe.vtool.lb import WeightedRoundRobin, parse_ratio


class LoadBalancerTests(unittest.TestCase):
    def test_parse_ratio(self):
        self.assertEqual(parse_ratio("3:1"), (3, 1))

    def test_weighted_round_robin_ratio_can_update_at_runtime(self):
        async def scenario():
            balancer = WeightedRoundRobin(
                endpoints=["http://127.0.0.1:8001", "http://127.0.0.1:8002"],
                weights=[2, 1],
            )
            initial = [await balancer.next() for _ in range(3)]
            self.assertEqual(
                initial,
                [
                    "http://127.0.0.1:8001",
                    "http://127.0.0.1:8001",
                    "http://127.0.0.1:8002",
                ],
            )

            await balancer.update_ratio("1:2")
            updated = [await balancer.next() for _ in range(3)]
            self.assertEqual(
                updated,
                [
                    "http://127.0.0.1:8001",
                    "http://127.0.0.1:8002",
                    "http://127.0.0.1:8002",
                ],
            )

            state = await balancer.get_state()
            self.assertEqual(state["ratio"], "1:2")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
