import pytest

import kornia as kornia
import kornia.testing as utils  # test utils

import torch
from torch.testing import assert_allclose
from torch.autograd import gradcheck


class TestCornerHarris:
    def test_shape(self):
        inp = torch.ones(1, 3, 4, 4)
        harris = kornia.feature.CornerHarris(k=0.04)
        assert harris(inp).shape == (1, 3, 4, 4)

    def test_shape_batch(self):
        inp = torch.zeros(2, 6, 4, 4)
        harris = kornia.feature.CornerHarris(k=0.04)
        assert harris(inp).shape == (2, 6, 4, 4)

    def test_corners(self):
        inp = torch.tensor([[[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]]).float()

        expected = torch.tensor([[[
            [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012],
            [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
            [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
            [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
            [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012]]]]).float()
        harris = kornia.feature.CornerHarris(k=0.04)
        scores = harris(inp)
        assert_allclose(scores, expected)

    def test_corners_batch(self):
        inp = torch.tensor([[
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ]]).repeat(2, 1, 1, 1)
        expected = torch.tensor([[[
            [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012],
            [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
            [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0020, 0.0040, 0.0029, 0.0000, 0.0029, 0.0040, 0.0020],
            [0.0039, 0.0065, 0.0040, 0.0000, 0.0040, 0.0065, 0.0039],
            [0.0012, 0.0039, 0.0020, 0.0000, 0.0020, 0.0039, 0.0012]
        ], [
            [0.001233, 0.003920, 0.001985, 0.000000, 0.001985, 0.003920, 0.001233],
            [0.003920, 0.006507, 0.003976, 0.000000, 0.003976, 0.006507, 0.003920],
            [0.001985, 0.003976, 0.002943, 0.000000, 0.002943, 0.003976, 0.001985],
            [0.001985, 0.003976, 0.002943, 0.000000, 0.002943, 0.003976, 0.001985],
            [0.003920, 0.006507, 0.003976, 0.000000, 0.003976, 0.006507, 0.003920],
            [0.000589, 0.001526, 0.000542, 0.000000, 0.000542, 0.001526, 0.000589],
            [0.000000, 0.000008, 0.000000, 0.000000, 0.000000, 0.000008, 0.000000]]
        ]])
        scores = kornia.feature.harris_response(inp, k=0.04)
        assert_allclose(scores, expected)

    def test_gradcheck(self):
        k = 0.04
        batch_size, channels, height, width = 1, 2, 5, 4
        img = torch.rand(batch_size, channels, height, width)
        img = utils.tensor_to_gradcheck_var(img)  # to var
        assert gradcheck(kornia.feature.harris_response, (img, k),
                         raise_exception=True)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self):
        @torch.jit.script
        def op_script(input, k):
            return kornia.feature.harris_response(input, k)
        k = torch.tensor(0.04)
        img = torch.rand(2, 3, 4, 5)
        actual = op_script(img, k)
        expected = kornia.feature.harris_response(img, k)
        assert_allclose(actual, expected)
