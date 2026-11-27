"""Command line defaults of the standalone mg7 gridpack launcher, taken from
its Cards/grid_run_card.toml."""

import argparse
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

_TEMPLATE = (Path(__file__).resolve().parents[3] / "madgraph" / "iolibs"
             / "template_files" / "mg7" / "gridpack.py")



def _run_card(**sections):
    """The default run card as plain TOML data, with sections replaced (None
    removes one)."""
    import madgraph.various.banner as banner_mod
    card = banner_mod.RunCardMG7()
    data = {key: dict(card[key])
            for key in ("run", "output", "generation", "gridpack", "amplicol")}
    data["generation"]["max_overweight_truncation"] = 0.001
    for key, value in sections.items():
        if value is None:
            del data[key]
        else:
            data[key] = value
    return data


class _Parsed(Exception):
    pass


def _defaults(run_card, argv=(), has_madspin=False):
    """The parsed arguments of main() with the command line ``argv``;
    ``has_madspin``: whether the gridpack carries MadSpin."""
    loader = types.ModuleType("custom_function_loader")
    setup = types.ModuleType("gridpack_setup")
    setup.__getattr__ = lambda name: None
    setup.load_run_card = lambda: run_card
    setup.load_madspace_data = dict
    spec = importlib.util.spec_from_file_location("mg7_gridpack_card", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"madspace": types.ModuleType("madspace"),
                                       "custom_function_loader": loader,
                                       "gridpack_setup": setup}):
        spec.loader.exec_module(module)
    module.has_madspin = lambda: has_madspin

    def parse_args(parser):
        raise _Parsed(parser.parse_known_args(list(argv))[0])

    with mock.patch.object(argparse.ArgumentParser, "parse_args", parse_args):
        try:
            module.main()
        except _Parsed as parsed:
            return parsed.args[0]
    raise AssertionError("main() did not parse the arguments")


class TestGridpackCard(unittest.TestCase):
    def test_card_without_amplicol_section(self):
        args = _defaults(_run_card(amplicol=None))
        self.assertEqual(args.first_pass_max_overweight_truncation, 0.001)
        self.assertEqual(args.first_pass_freeze_max_weight_after, 10000)
        self.assertEqual(args.max_overweight_truncation, 0.001)

    def test_amplicol_section(self):
        args = _defaults(_run_card(amplicol={"first_pass_max_overweight_truncation": 0.2}))
        self.assertEqual(args.first_pass_max_overweight_truncation, 0.2)
        self.assertEqual(args.first_pass_freeze_max_weight_after, 10000)


class TestGridpackMadSpinOptions(unittest.TestCase):
    """--madspin / --no_madspin / --keep_undecayed of a gridpack that may carry
    MadSpin ([madspin] enable of its grid run card)."""

    def test_default_follows_the_card(self):
        card = _run_card(madspin={"enable": True})
        self.assertTrue(_defaults(card, has_madspin=True).madspin)
        self.assertFalse(_defaults(card, has_madspin=False).madspin)
        self.assertFalse(_defaults(_run_card(), has_madspin=True).madspin)
        self.assertFalse(_defaults(_run_card(madspin={"enable": False}),
                                   has_madspin=True).madspin)

    def test_command_line(self):
        card = _run_card(madspin={"enable": True})
        self.assertFalse(_defaults(card, ["--no_madspin"], True).madspin)
        self.assertTrue(_defaults(_run_card(), ["--madspin"], True).madspin)
        args = _defaults(card, ["--keep_undecayed"], True)
        self.assertTrue(args.keep_undecayed)
        self.assertFalse(_defaults(card, [], True).keep_undecayed)


if __name__ == "__main__":
    unittest.main()
