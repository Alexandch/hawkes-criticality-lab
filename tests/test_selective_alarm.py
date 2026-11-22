from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from critical_processes.selective_alarm import fit_sector_calibration,apply_sector_calibration,fit_selective_policy,apply_selective_policy,choose


class SelectiveAlarmTests(unittest.TestCase):
    def test_sector_intercepts_capture_different_base_rates(self):
        rng=np.random.default_rng(11);score=np.full((2000,2),.2);labels=rng.random((2000,2))<np.array([.08,.35])
        record=fit_sector_calibration(score,labels);p=apply_sector_calibration(score,record)
        self.assertGreater(p[:,1].mean(),p[:,0].mean()+.15)

    def test_positive_slope_preserves_ranking(self):
        x=np.linspace(.01,.99,1000)[:,None];y=x>.7;record=fit_sector_calibration(x,y)
        self.assertTrue(np.all(np.diff(apply_sector_calibration(x,record)[:,0])>0))

    def test_constant_label_fallback_is_finite(self):
        x=np.ones((100,2))*.1;record=fit_sector_calibration(x,np.zeros_like(x))
        self.assertTrue(record['fallback']);self.assertTrue(np.isfinite(apply_sector_calibration(x,record)).all())

    def test_risk_floor_abstains_without_modifying_probability(self):
        raw=np.full((3,2,1),.8);policy=dict(kind='sector',probability=[dict(slope=1.,intercept=-4.,offset=[0.,0.])],raw_threshold=[[.5],[.5]],risk_floor=.3)
        p,alarm=apply_selective_policy(raw,policy);self.assertTrue((p<.3).all());self.assertFalse(alarm.any())

    def test_post_asof_and_immature_labels_do_not_change_policy(self):
        dates=pd.date_range('2024-01-01',periods=400*4,freq='6h');rng=np.random.default_rng(4)
        raw=rng.uniform(.01,.7,(len(dates),2,1));cut=np.ones_like(raw)*2;known=rng.poisson(.8,raw.shape)
        first=fit_selective_policy(raw,cut,known,dates,'2024-12-01');changed=known.copy();changed[dates>=pd.Timestamp('2024-12-01')-pd.Timedelta(days=56)]=100
        second=fit_selective_policy(raw,cut,changed,dates,'2024-12-01');self.assertEqual(first,second)

    def test_tied_raw_threshold_is_conservative(self):
        dates=pd.date_range('2024-01-01',periods=400*4,freq='6h');raw=np.full((len(dates),2,1),.5);cut=np.ones_like(raw);known=np.zeros_like(raw)
        policy=fit_selective_policy(raw,cut,known,dates,'2024-12-01',risk_floor=0.)
        self.assertFalse(apply_selective_policy(raw[:3],policy)[1].any())

    def test_selection_uses_all_quarters(self):
        good=dict(precision=.35,recall=.12,fpr=.04,fbeta_half=.25);bad={**good,'fpr':.10}
        rows=[dict(id='unstable',method='nb_plain',policy='monthly',overall={**good,'precision':.9},quarters=[good,good,good,bad]),
              dict(id='stable',method='nb_plain',policy='weekly_sector',overall=good,quarters=[good]*4)]
        self.assertEqual(choose(rows)['id'],'stable')


if __name__=='__main__':unittest.main()
