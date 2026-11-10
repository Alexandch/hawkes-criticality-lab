from __future__ import annotations
import contextlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.spatiotemporal import (DistrictPanel,district_graph,fit_calendar,calendar_prediction,forecast_features,exponential_history,
    average_precision,fit_hawkes_count,poisson_nll,observed_rolling,observed_memory,counts_available_by)
from critical_processes.spatial_experiment import alarm_threshold,run_spatial_experiment
from critical_processes.selected_training import sha
from critical_processes.district_forecast import forecast_interval


def synthetic_panel():
    dates=pd.date_range('2019-01-01','2026-10-01',freq='24h',inclusive='left');rng=np.random.default_rng(331)
    means=np.array([2.,4.,7.])[None,:]*(1+.15*np.sin(np.arange(len(dates))[:,None]*2*np.pi/7))
    return DistrictPanel(rng.poisson(means),dates.to_numpy(dtype='datetime64[ns]').astype(np.int64),np.array(['A','B','C']),np.array([[41.,-87.],[41.05,-87.02],[41.03,-87.1]]),24)


class SpatiotemporalTests(unittest.TestCase):
    def test_late_reports_are_not_visible_before_registration(self):
        dates=pd.date_range('2020-01-01',periods=10,freq='D');counts=np.zeros((10,2),dtype=int)
        counts[0,0]=1;counts[1,1]=1
        panel=DistrictPanel(counts,dates.to_numpy(dtype='datetime64[ns]').astype(np.int64),np.array(['a','b']),np.array([[1.,2.],[2.,3.]]),24,
            event_bins=np.array([0,1]),report_bins=np.array([4,1]),area_indices=np.array([0,1]))
        observed=observed_rolling(panel,7);self.assertEqual(observed[:5,0].sum(),0);self.assertEqual(observed[5,0],1)
        self.assertEqual(counts_available_by(panel,'2020-01-05').sum(),1)
        memory=observed_memory(panel,3);self.assertEqual(memory[:5,0].sum(),0)
        self.assertAlmostEqual(memory[5,0],(1-np.exp(-1/3))*np.exp(-4/3))

    def test_graph_has_no_self_edges_and_preserves_probability_rows(self):
        graph=district_graph(synthetic_panel().centroids)
        np.testing.assert_array_equal(np.diag(graph),0.);np.testing.assert_allclose(graph.sum(1),1.)
        self.assertTrue((graph>=0).all())

    def test_calendar_does_not_fit_to_future_counts(self):
        panel=synthetic_panel();first=fit_calendar(panel)
        panel.counts[panel.dates>=pd.Timestamp('2024-01-01')]=1000
        self.assertEqual(first,fit_calendar(panel))

    def test_history_and_features_use_only_available_past(self):
        panel=synthetic_panel();calendar=fit_calendar(panel);base,_=calendar_prediction(calendar,panel.dates);graph=district_graph(panel.centroids)
        first=forecast_features(panel,base,graph,7);memory=exponential_history(panel.counts,3,7)
        cutoff=2000;panel.counts[cutoff:]=10000
        second=forecast_features(panel,base,graph,7);new_memory=exponential_history(panel.counts,3,7)
        np.testing.assert_array_equal(first[:cutoff+8],second[:cutoff+8]);np.testing.assert_array_equal(memory[:cutoff+8],new_memory[:cutoff+8])

    def test_equal_score_average_precision_is_prevalence(self):
        labels=np.array([1,0,0,1],dtype=bool)
        self.assertAlmostEqual(average_precision(labels,np.ones(4)),.5)
        self.assertAlmostEqual(average_precision(labels,np.array([.8,.1,.2,.9])),1.)

    def test_calibration_strict_ties_do_not_create_unbudgeted_alarms(self):
        values=np.r_[np.full(95,.1),np.full(5,.9)];labels=np.zeros(100,dtype=bool)
        threshold=alarm_threshold(values,labels)
        self.assertLessEqual((values>=threshold).mean(),.05)
        self.assertGreater(threshold,.1)

    def test_spatial_mle_is_stable_and_can_match_background(self):
        panel=synthetic_panel();base=np.broadcast_to(panel.counts.mean(0),panel.counts.shape).copy();memory=exponential_history(panel.counts,3)
        mask=np.ones(len(base),dtype=bool);graph=district_graph(panel.centroids)
        predicted,record=fit_hawkes_count(panel.counts,base,memory,graph,mask,True)
        self.assertLessEqual(record['spectral_radius_proxy'],.990001)
        self.assertLessEqual(poisson_nll(panel.counts,predicted).mean(),poisson_nll(panel.counts,base).mean()+1e-7)

    def test_complete_new_target_freezes_selection_and_repeats_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);panels=root/'panels';out=root/'result';panels.mkdir();panel=synthetic_panel()
            for city in ('chicago','san_francisco'):
                np.savez_compressed(panels/(city+'.npz'),counts=panel.counts,time_ns=panel.time_ns,districts=panel.districts,centroids=panel.centroids)
                (panels/(city+'.json')).write_text(json.dumps({'city':city}),encoding='utf-8')
            kwargs=dict(panel_dir=panels,out=out,epochs=2,model_seeds=(5,),bin_hours=24,bootstrap_samples=5)
            with contextlib.redirect_stdout(io.StringIO()):first=run_spatial_experiment(**kwargs)
            self.assertEqual(len(first),40)
            frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'));started=json.loads((out/'test_started.json').read_text(encoding='utf-8'))
            self.assertFalse(frozen['test_used_for_selection']);self.assertEqual(started['frozen_plan_sha256'],sha(out/'frozen_plan.json'))
            before=sha(out/'test_metrics.csv');second=run_spatial_experiment(**kwargs)
            pd.testing.assert_frame_equal(first,second,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
            self.assertEqual(before,sha(out/'test_metrics.csv'))
            next_interval=forecast_interval('chicago',out,panels)
            self.assertEqual(len(next_interval),3);self.assertTrue(next_interval.observed_count.isna().all())
            self.assertTrue((next_interval.expected_count>0).all());self.assertEqual(next_interval.forecast_start.iloc[0],'2026-10-01 00:00:00')
            with self.assertRaises(ValueError):forecast_interval('chicago',out,panels,asof='2026-10-01T00:00:00Z')
            with self.assertRaises(ValueError):run_spatial_experiment(**{**kwargs,'epochs':3})
            original=(panels/'chicago.npz').read_bytes()
            with (panels/'chicago.npz').open('ab') as handle:handle.write(b'changed-panel')
            with self.assertRaises(ValueError):forecast_interval('chicago',out,panels)
            (panels/'chicago.npz').write_bytes(original)


if __name__=='__main__':unittest.main()
