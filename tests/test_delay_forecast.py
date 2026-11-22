from __future__ import annotations
import importlib.util,sys,unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from critical_processes.delay_forecast import fit_reporting_cdf,effective_exposure,corrected_ratio,nb2_nll,build_network,predict_network
from critical_processes.offline_forecast import CategoryPanel
from critical_processes.spatiotemporal import past_rolling


def prepare_module():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_delay_forecast_data.py';sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('r14_data_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class DelayForecastTests(unittest.TestCase):
    def test_complete_reporting_exposure_equals_complete_past_window(self):
        base=np.random.default_rng(9).uniform(.1,5,size=(100,2));cdf=np.ones((2,100))
        np.testing.assert_allclose(effective_exposure(base,cdf,8,4),past_rolling(base,8,4),atol=1e-12)

    def test_no_publication_information_keeps_ratio_neutral(self):
        exposure=effective_exposure(np.ones((40,2)),np.zeros((2,100)),8,4)
        self.assertFalse(exposure.any());np.testing.assert_array_equal(corrected_ratio(np.zeros_like(exposure),exposure,1.),1.)

    def test_partial_reporting_corrects_missing_exposure_instead_of_zero_crime(self):
        complete=np.ones((30,2))*20;observed=complete*.4;expected=complete*.4
        np.testing.assert_array_equal(corrected_ratio(observed,expected,2.),1.)
        self.assertTrue((corrected_ratio(observed,complete,2.)<1).all())

    def test_delay_profile_excludes_future_cohorts_and_is_monotone(self):
        dates=pd.date_range('2021-04-01','2025-01-01',freq='6h',inclusive='left').as_unit('ns');events=np.array([10,20,30,40])
        panel=CategoryPanel(np.zeros((len(dates),2,3),dtype=int),dates.asi8,np.array(['a','b']),np.array(['all','property','assault_robbery']),np.zeros((2,2)),events,np.array([0,0,1,1]),np.array([1,2,1,2]),events+np.array([0,4,8,12]))
        first,_=fit_reporting_cdf(panel,max_age=100)
        future=int(np.searchsorted(dates.asi8,pd.Timestamp('2024-02-01').value))
        changed=replace(panel,event_bins=np.append(events,future),area_indices=np.append(panel.area_indices,0),group_indices=np.append(panel.group_indices,1),report_bins=np.append(panel.report_bins,future))
        second,_=fit_reporting_cdf(changed,max_age=100)
        np.testing.assert_array_equal(first,second);self.assertTrue((np.diff(first,axis=2)>=0).all());self.assertFalse(first[:,:,:29].any())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch missing')
    def test_nb2_likelihood_matches_scipy_and_has_finite_gradients(self):
        import torch
        y=np.array([0.,1.,5.,20.]);mu=np.array([.2,2.,5.,8.]);alpha=np.array([.01,.2,1.,3.])
        logmean=torch.tensor(np.log(mu),requires_grad=True);disp=torch.tensor(alpha,requires_grad=True)
        result=nb2_nll(torch.tensor(y),logmean,disp)
        np.testing.assert_allclose(result.detach().numpy(),-nbinom.logpmf(y,1/alpha,1/(1+alpha*mu)),rtol=1e-10,atol=1e-10)
        result.sum().backward();self.assertTrue(torch.isfinite(logmean.grad).all());self.assertTrue(torch.isfinite(disp.grad).all())

    def test_local_hhmm_does_not_invent_invalid_times(self):
        module=prepare_module();result=module.local_clock(pd.Series(['2023-01-01']*4),pd.Series(['407','2359','1265','2400']))
        self.assertEqual(result.iloc[0],pd.Timestamp('2023-01-01 04:07'));self.assertEqual(result.iloc[1],pd.Timestamp('2023-01-01 23:59'));self.assertTrue(result.iloc[2:].isna().all())

    def test_forecast_lead_is_past_even_across_dst_and_later_bins_do_not_change_it(self):
        module=prepare_module();dates=pd.DatetimeIndex(['2024-11-03 00:00','2024-11-03 06:00']).as_unit('ns')
        hours=pd.date_range('2024-11-03 05:00',periods=14,freq='h',tz='UTC');series=pd.Series(np.arange(14.),index=hours)
        first,issue=module.weather_features(series,dates)
        self.assertEqual(first[0,0],3.);self.assertEqual(first[0,1],6.)
        self.assertLess(issue[0],dates[0].tz_localize('America/Chicago').tz_convert('UTC').value-pd.Timedelta(hours=12).value)
        series.iloc[7:]+=1000;second,_=module.weather_features(series,dates);np.testing.assert_array_equal(first[0],second[0])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch missing')
    def test_distribution_network_produces_finite_positive_mean_and_dispersion(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(13);network=build_network(4,2,'nb')
        record=dict(width=4,areas=2,kind='nb',state_dict=network.state_dict(),mean=np.zeros(4,dtype='float32'),scale=np.ones(4,dtype='float32'))
        x=np.zeros((10,2,3,4),dtype='float32');base=np.ones((10,2,3))
        mean,alpha=predict_network(x,base,record)
        self.assertTrue(np.isfinite(mean).all() and (mean>0).all());self.assertTrue(((alpha>=.001)&(alpha<=5.001)).all())


if __name__=='__main__':unittest.main()
