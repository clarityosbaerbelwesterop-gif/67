"""DiLoCo engine: outer step, codecs and their byte counts."""

import unittest

import torch

from fabric.diloco import Codec, Outer, assign, average_payloads, flatten, outer_gradient


class OuterTest(unittest.TestCase):
    def test_lr_one_without_momentum_is_plain_averaging(self):
        theta = torch.zeros(6)
        workers = [torch.arange(6.0), torch.arange(6.0) * 3]
        outer = Outer(theta, lr=1.0, momentum=0.0)
        grad, _ = outer_gradient(outer.theta, workers, [Codec("fp32"), Codec("fp32")])
        torch.testing.assert_close(outer.step(grad), torch.arange(6.0) * 2)

    def test_nesterov_matches_torch_sgd(self):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(5))
        ref = torch.optim.SGD([p], lr=0.7, momentum=0.9, nesterov=True)
        outer = Outer(p.detach().clone(), lr=0.7, momentum=0.9)
        for _ in range(3):
            g = torch.randn(5)
            p.grad = g.clone()
            ref.step()
            outer.step(g)
        torch.testing.assert_close(outer.theta, p.detach())

    def test_state_round_trip(self):
        outer = Outer(torch.ones(4))
        outer.step(torch.full((4,), 0.5))
        again = Outer.load(outer.state())
        torch.testing.assert_close(again.step(torch.ones(4)), outer.step(torch.ones(4)))


class CodecTest(unittest.TestCase):
    def test_bytes_on_the_wire(self):
        delta = torch.randn(10_000)
        self.assertEqual(Codec("fp32").encode(delta)[1], 40_000)
        self.assertEqual(Codec("fp16").encode(delta)[1], 20_000)
        self.assertEqual(Codec("int8", chunk=4096).encode(delta)[1], 3 * 4096 + 3 * 4)
        self.assertEqual(Codec("topk:0.01").encode(delta)[1], 100 * 6)

    def test_int8_error_is_at_most_half_a_step(self):
        delta = torch.randn(9_000)
        codec = Codec("int8", chunk=1000)
        payload, _ = codec.encode(delta)
        err = (Codec.decode(payload) - delta).abs()
        step = payload["scale"].repeat_interleave(1000)[:9000]
        self.assertTrue(bool((err <= step / 2 + 1e-7).all()))

    def test_topk_error_feedback_loses_nothing_over_rounds(self):
        torch.manual_seed(1)
        codec = Codec("topk:0.05")
        sent, total = torch.zeros(500), torch.zeros(500)
        for _ in range(10):
            delta = torch.randn(500)
            total += delta
            payload, _ = codec.encode(delta)
            sent += Codec.decode(payload)
        # what was sent plus what is still held back equals everything that was produced (fp16 rounding aside)
        torch.testing.assert_close(sent + codec.residual, total, atol=2e-2, rtol=0)

    def test_assign_flatten_round_trip(self):
        model = torch.nn.Linear(3, 2)
        vec = torch.arange(8.0)
        assign(list(model.parameters()), vec)
        torch.testing.assert_close(flatten(model.parameters()), vec)

    def test_unknown_codec_is_refused(self):
        with self.assertRaises(ValueError):
            Codec("zip")


class QuorumTest(unittest.TestCase):
    def test_a_round_survives_lost_workers_down_to_the_quorum(self):
        payloads = [Codec("fp32").encode(torch.full((3,), float(v)))[0] for v in (1, 2, 3)]
        grad, used = average_payloads(payloads, expected=4, min_workers=3)
        self.assertEqual(used, 3)
        torch.testing.assert_close(grad, torch.full((3,), 2.0))
        with self.assertRaises(SystemExit):
            average_payloads(payloads[:2], expected=4, min_workers=3)
        with self.assertRaises(ValueError):
            average_payloads(payloads * 2, expected=4, min_workers=3)


if __name__ == "__main__":
    unittest.main()
