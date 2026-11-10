import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from critical_processes.extended_training import fused_features, run_extended_training
from critical_processes.selected_training import sha


class ExtendedTrainingTests(unittest.TestCase):
    def test_fusion_selects_same_90_and_360_blocks_across_old_schemas(self):
        old=np.zeros((1,4,10,64));old[:,0]=30.;old[:,1]=90.;old[:,2]=180.;old[:,3]=360.
        new=np.zeros((1,3,10,64));new[:,0]=90.;new[:,1]=180.;new[:,2]=360.
        np.testing.assert_array_equal(fused_features(SimpleNamespace(matrix=old),[30,90,180,360]),
                                      fused_features(SimpleNamespace(matrix=new),[90,180,360]))
        self.assertEqual(fused_features(SimpleNamespace(matrix=new),[90,180,360]).shape,(1,10,128))

    def test_new_holdout_is_frozen_and_completed_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs=dict(out=directory,new_train_per_family=2,validation_per_family=2,calibration_per_family=19,test_per_family=2,
                epochs=3,model_seeds=(11,13),data_seed=16231,test_seed=16232,workers=1,bootstrap_samples=5,
                prefix_days=60,stage_days=4,stages=8,rate_min=.5,rate_max=2.,sources=[])
            with contextlib.redirect_stdout(io.StringIO()): result=run_extended_training(**kwargs)
            out=Path(directory)
            self.assertEqual(len(result),18)
            frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
            started=json.loads((out/'test_started.json').read_text(encoding='utf-8'))
            self.assertFalse(frozen['test_used_for_selection'])
            self.assertEqual(started['frozen_plan_sha256'],sha(out/'frozen_plan.json'))
            before=sha(out/'data/test.npz')
            self.assertEqual(result,run_extended_training(**kwargs))
            self.assertEqual(before,sha(out/'data/test.npz'))
            with self.assertRaises(ValueError):run_extended_training(**{**kwargs,'epochs':4})


if __name__=='__main__':unittest.main()
