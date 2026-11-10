from __future__ import annotations
import contextlib,io,json,sqlite3,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.offline_forecast import (CategoryPanel,vintage_masks,available_targets,causal_inputs,rise_cutoff,
    calibrate_probability,apply_calibration,regional_thresholds,evaluate_forecast)
from critical_processes.offline_experiment import run_offline
from critical_processes.observation_ledger import record_snapshot


def synthetic():
    dates=pd.date_range('2019-01-01','2026-11-01',freq='6h',inclusive='left');rng=np.random.default_rng(239)
    parts=rng.poisson(np.array([[2.,1.],[1.,.4]])[None,:,:],size=(len(dates),2,2))
    counts=np.concatenate([parts.sum(2)[:,:,None],parts],axis=2)
    return CategoryPanel(counts,dates.to_numpy(dtype='datetime64[ns]').astype('int64'),np.array(['a','b']),np.array(['all','property','assault_robbery']),
        np.array([[40.,-70.],[40.1,-70.1]]),np.array([],dtype=int),np.array([],dtype=int),np.array([],dtype=int))


class OfflineReadinessTests(unittest.TestCase):
    def test_rise_is_relative_to_available_background_with_absolute_floor(self):
        np.testing.assert_array_equal(rise_cutoff(np.array([.2,2.,8.])),[3,4,12])

    def test_vintage_maturity_and_calibration_separation(self):
        panel=synthetic();masks=vintage_masks(panel.dates,'2026-10-01')
        self.assertFalse(np.any(masks['train']&masks['validation']))
        self.assertFalse(np.any(masks['probability_calibration']&masks['alarm_calibration']))
        self.assertLess(panel.dates[masks['alarm_calibration']][-1],pd.Timestamp('2026-09-03'))
        self.assertLess(panel.dates[masks['train']][-1],pd.Timestamp('2026-03-04'))

    def test_future_outcomes_do_not_change_causal_inputs(self):
        panel=synthetic();first=causal_inputs(panel,'2025-07-01');cut=np.searchsorted(panel.time_ns,pd.Timestamp('2026-10-01').value)
        panel.counts[cut:,:,1:]+=100;panel.counts[cut:,:,0]=panel.counts[cut:,:,1:].sum(2)
        second=causal_inputs(panel,'2025-07-01')
        for index in range(3):np.testing.assert_array_equal(first[index][:cut],second[index][:cut])

    def test_report_registration_plus_publication_delay_controls_targets(self):
        dates=pd.date_range('2026-01-01',periods=20,freq='6h');counts=np.zeros((20,2,3),dtype=int);counts[0,0,[0,1]]=1
        panel=CategoryPanel(counts,dates.to_numpy(dtype='datetime64[ns]').astype('int64'),np.array(['a','b']),np.array(['all','property','assault_robbery']),
            np.array([[40.,-70.],[40.1,-70.1]]),np.array([0]),np.array([0]),np.array([1]),np.array([2]))
        self.assertEqual(available_targets(panel,'2026-01-04',publication_days=3).sum(),0)
        self.assertEqual(available_targets(panel,'2026-01-05',publication_days=3).sum(),2)

    def test_probability_calibration_is_monotone_and_handles_single_class(self):
        scores=np.linspace(.01,.99,100);labels=scores>.6;fit=calibrate_probability(scores,labels)
        self.assertTrue((np.diff(apply_calibration(scores,fit))>=0).all())
        constant=calibrate_probability(scores,np.zeros(100,dtype=bool));self.assertTrue(constant['fallback'])
        self.assertGreater(apply_calibration(.5,constant),0)

    def test_district_thresholds_handle_ties_and_small_samples(self):
        scores=np.ones((100,2,3))*.1;labels=np.zeros_like(scores,dtype=bool)
        thresholds,notes=regional_thresholds(scores,labels);self.assertTrue((thresholds>.1).all());self.assertFalse(np.array(notes['fallback']).any())
        _,small=regional_thresholds(scores[:10],labels[:10]);self.assertTrue(np.array(small['fallback']).all())

    def test_confusion_counts_and_precision_use_alarm_denominator(self):
        y=np.array([[8,0],[0,8]]);mean=np.ones_like(y)*3;prob=np.array([[.9,.8],[.1,.2]])
        metric=evaluate_forecast(y,mean,prob,np.ones_like(y)*6,np.ones(2)*.5,.1)
        self.assertEqual(metric['true_positives'],1);self.assertEqual(metric['false_positives'],1)
        self.assertEqual(metric['precision'],.5);self.assertEqual(metric['recall'],.5)

    def test_first_seen_is_actual_observation_and_missing_partial_scope_is_not_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observations.sqlite';old=[('1',{'incident':'2020-01-01'}),('2',{'x':1})]
            first=record_snapshot(path,'city',old,'2026-11-10T10:00:00Z','full')
            self.assertEqual(first['new'],2);self.assertTrue(record_snapshot(path,'city',old[::-1],'2026-11-10T10:00:00Z','full')['cached'])
            changed=record_snapshot(path,'city',[('1',{'incident':'2020-01-02'})],'2026-11-11T10:00:00Z','partial')
            self.assertEqual(changed['changed'],1)
            with contextlib.closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM observations').fetchone()[0],2)
                self.assertTrue(db.execute('SELECT first_seen FROM observations WHERE record_id="1"').fetchone()[0].startswith('2026-11-10'))
            with self.assertRaises(ValueError):record_snapshot(path,'city',old,'2026-11-09T10:00:00Z','older')

    def test_full_offline_run_freezes_predictions_without_outcomes_and_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir();panel=synthetic()
            for city in ('chicago','san_francisco'):
                np.savez_compressed(data/(city+'.npz'),**{name:value for name,value in vars(panel).items() if value is not None})
                (data/(city+'.json')).write_text('{}',encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):result=run_offline(data,root/'result',epochs=2,seeds=(5,),vintages=('2026-10-01',),bootstrap=10)
            self.assertEqual(len(result),96)
            journal=json.loads((root/'result/forecast_journal.json').read_text(encoding='utf-8'));self.assertFalse(journal['contains_outcomes'])
            self.assertEqual(len(journal['entries']),4)
            with np.load(root/'result'/journal['entries'][0]['file']) as saved:self.assertNotIn('counts',saved.files)
            for city in ('chicago','san_francisco'):
                entries=[entry for entry in journal['entries'] if entry['city']==city]
                with np.load(root/'result'/entries[0]['file']) as first,np.load(root/'result'/entries[1]['file']) as second:
                    np.testing.assert_array_equal(first['cutoff'],second['cutoff'])
            again=run_offline(data,root/'result',epochs=2,seeds=(5,),vintages=('2026-10-01',),bootstrap=10)
            pd.testing.assert_frame_equal(result,again,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
            readiness=json.loads((root/'result/readiness.json').read_text(encoding='utf-8'));self.assertFalse(readiness['prospective_validation_performed'])


if __name__=='__main__':unittest.main()
