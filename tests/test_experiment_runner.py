"""Guard against silently accepting incomplete or stale cached experiments."""

import json
from pathlib import Path
import tempfile
import unittest

from run_experiments import check_completed_run, grid
from train_authorship_attribution import ROOT


class ExperimentRunnerTests(unittest.TestCase):
    def test_grid_contains_fourteen_distinct_runs(self):
        names = [run[0] for run in grid()]
        self.assertEqual(len(names), 14)
        self.assertEqual(len(set(names)), 14)

    def test_missing_evidence_is_rejected(self):
        source = ROOT / "results/cnn_output_s42/metrics.json"
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            (destination / "metrics.json").write_bytes(source.read_bytes())
            with self.assertRaises(FileNotFoundError):
                check_completed_run(destination, ("cnn", "output", "masked", 42, False))

    def test_existing_published_run_is_accepted(self):
        check_completed_run(
            ROOT / "results/cnn_output_s42", ("cnn", "output", "masked", 42, False)
        )
