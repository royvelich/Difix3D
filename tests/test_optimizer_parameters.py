import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_difix import (  # noqa: E402
    _build_gram_feature_network,
    _optimizer_parameters,
)


class _FakeLoraSkip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.ones(1))
        self.lora_A = torch.nn.ModuleDict(
            {"vae_skip": torch.nn.Linear(1, 1, bias=False)}
        )


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        for index in range(1, 5):
            setattr(self.decoder, f"skip_conv_{index}", _FakeLoraSkip())
        self.other_lora = torch.nn.ModuleDict(
            {"vae_skip": torch.nn.Linear(1, 1, bias=False)}
        )
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class _FakeDifix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = torch.nn.Linear(1, 1)
        self.vae = _FakeVAE()


class OptimizerParametersTest(unittest.TestCase):
    def test_zero_gram_weight_does_not_construct_vgg(self):
        with mock.patch("train_difix.torchvision.models.vgg16") as vgg16:
            network = _build_gram_feature_network(0)

        self.assertIsNone(network)
        vgg16.assert_not_called()

    def test_positive_gram_weight_constructs_frozen_eval_vgg(self):
        features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=1)
        )
        vgg = mock.Mock(features=features)
        with mock.patch(
            "train_difix.torchvision.models.vgg16",
            return_value=vgg,
        ) as vgg16:
            network = _build_gram_feature_network(1)

        self.assertIs(network, features)
        self.assertFalse(network.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in network.parameters())
        )
        vgg16.assert_called_once_with(pretrained=True)

    def test_default_contains_each_trainable_parameter_once(self):
        model = _FakeDifix()

        parameters = _optimizer_parameters(model)

        expected = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(
            {id(parameter) for parameter in parameters},
            {id(parameter) for parameter in expected},
        )
        self.assertEqual(
            len(parameters),
            len({id(parameter) for parameter in parameters}),
        )

    def test_legacy_mode_duplicates_only_skip_lora_parameters(self):
        model = _FakeDifix()

        parameters = _optimizer_parameters(
            model,
            legacy_difix_optimizer=True,
        )
        reference_counts = {}
        for parameter in parameters:
            reference_counts[id(parameter)] = (
                reference_counts.get(id(parameter), 0) + 1
            )

        skip_lora_ids = {
            id(
                getattr(model.vae.decoder, f"skip_conv_{index}")
                .lora_A["vae_skip"]
                .weight
            )
            for index in range(1, 5)
        }
        duplicated_ids = {
            parameter_id
            for parameter_id, count in reference_counts.items()
            if count == 2
        }

        self.assertEqual(duplicated_ids, skip_lora_ids)
        self.assertTrue(
            all(count in (1, 2) for count in reference_counts.values())
        )
        self.assertEqual(
            reference_counts[id(model.vae.other_lora["vae_skip"].weight)],
            1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        optimizer_references = optimizer.param_groups[0]["params"]
        self.assertEqual(len(optimizer_references), len(parameters))


if __name__ == "__main__":
    unittest.main()
