from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from critical_processes.expanded_forecast import (chronological_masks,quarterly_release_bins,delayed_weather,
    fit_neural,predict_neural,operating_intervals,practical_screen)
from critical_processes.expanded_experiment import freeze_forecasts,METHODS,SCENARIOS
from critical_processes.spatiotemporal import DistrictPanel,observed_rolling,past_rolling


class ExpandedDetectorTests(unittest.TestCase):
    def test_nyc_publication_is_after_report_quarter_and_keeps_last_day_in_correct_quarter(self):
        origin=pd.Timestamp('2024-01-01').value
        dates=pd.DatetimeIndex(['2024-04-01','2024-04-02']).as_unit('ns')
        reports=(dates.asi8-origin)//(6*3600*10**9)
        releases=quarterly_release_bins(reports,origin,45)
        actual=pd.DatetimeIndex((origin+releases.astype('int64')*6*3600*10**9).astype('datetime64[ns]'))
        self.assertEqual(actual[0],pd.Timestamp('2024-05-16'))
        self.assertEqual(actual[1],pd.Timestamp('2024-08-15'))

    def test_nyc_alarm_calibration_excludes_unpublished_fourth_quarter(self):
        dates=pd.date_range('2024-01-01','2025-01-01',freq='6h',inclusive='left')
        masks=chronological_masks(dates,'nyc')
        self.assertEqual(dates[masks['alarm_calibration']][-1],pd.Timestamp('2024-09-30 18:00'))

    def test_quarterly_recent_window_uses_published_history_instead_of_missing_zeros(self):
        dates=pd.date_range('2024-01-01',periods=365*4,freq='6h').as_unit('ns');events=np.arange(len(dates))
        reports=(events//4+1)*4;releases=quarterly_release_bins(reports,dates[0].value,45)
        panel=DistrictPanel(np.ones((len(dates),1)),dates.asi8,np.array(['a']),np.array([[40.,-70.]]),6,events,releases,np.zeros(len(events),dtype=int))
        delay=SCENARIOS['primary']['nyc_lag_days']*4
        observed=observed_rolling(panel,28*4,delay);expected=past_rolling(panel.counts,28*4,delay)
        np.testing.assert_array_equal(observed[200*4],expected[200*4])
        self.assertGreater(observed[200*4,0],0)
        self.assertEqual(observed_rolling(panel,28*4,7*4)[200*4,0],0)

    def test_chronological_roles_and_maturity_are_disjoint(self):
        dates=pd.date_range('2019-01-01','2026-10-01',freq='6h',inclusive='left');masks=chronological_masks(dates)
        self.assertTrue((np.stack(list(masks.values())).sum(0)<=1).all())
        self.assertEqual(dates[masks['train']][-1],pd.Timestamp('2023-12-03 18:00'))
        self.assertEqual(dates[masks['alarm_calibration']][-1],pd.Timestamp('2024-12-03 18:00'))
        self.assertEqual(dates[masks['test2025']][0],pd.Timestamp('2025-01-01'))

    def test_weather_is_delayed_and_future_changes_do_not_rewrite_past_features(self):
        days=pd.date_range('2024-01-01',periods=20).as_unit('ns');dates=pd.date_range('2024-01-01',periods=80,freq='6h')
        values=np.arange(80,dtype=float).reshape(20,4);first=delayed_weather(dates,days.asi8,values,5)
        self.assertTrue((first[:20,4:]==1).all());self.assertEqual(first[20,0],values[0,0])
        values[10:]+=10000;second=delayed_weather(dates,days.asi8,values,5)
        np.testing.assert_array_equal(first[:60],second[:60])
        with self.assertRaises(ValueError):delayed_weather(dates,days.asi8,values,0)

    def test_weather_does_not_fill_missing_values_from_a_future_day(self):
        days=pd.date_range('2024-01-01',periods=4).as_unit('ns');values=np.ones((4,4));values[0,0]=np.nan
        result=delayed_weather(pd.date_range('2024-01-02',periods=4,freq='6h'),days.asi8,values,1)
        self.assertEqual(result[0,0],0);self.assertEqual(result[0,4],1)

    def test_noaa_quality_flags_missing_days_and_units(self):
        path=Path(__file__).resolve().parents[1]/'scripts/prepare_expanded_districts.py'
        import sys
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('expanded_preparation_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def line(element,value,quality=' '):return 'USW00094728'+'202401'+element+f'{value:5d} {quality} '+('-9999   '*30)
        result=module.parse_weather('\n'.join([line('TMAX',123),line('PRCP',25),line('TMIN',-40,'X')]),'USW00094728','2024-01-01','2024-02-01')
        self.assertEqual(result.loc['2024-01-01','TMAX'],12.3);self.assertEqual(result.loc['2024-01-01','PRCP'],2.5)
        self.assertTrue(pd.isna(result.loc['2024-01-01','TMIN']))

    def test_operating_intervals_preserve_all_districts_in_time_blocks(self):
        labels=np.tile(np.array([[1,0]],dtype=bool),(112,1));alarms=np.ones_like(labels);prob=np.full(labels.shape,.5)
        result=operating_intervals(labels,alarms,prob,prob,50)
        self.assertEqual(result['blocks'],2);self.assertIsNone(result['precision_ci_low'])
        result=operating_intervals(np.tile(labels,(2,1)),np.tile(alarms,(2,1)),np.tile(prob,(2,1)),np.tile(prob,(2,1)),50)
        self.assertEqual(result['precision_ci_low'],.5);self.assertEqual(result['brier_difference_ci_high'],0.)

    def test_practical_screen_requires_all_predeclared_conditions(self):
        row=dict(precision=.4,recall=.2,fpr=.04,ap=.3,baseline_ap=.2,brier_difference_ci_high=-.001)
        self.assertTrue(practical_screen(row)['passed']);row['precision']=.29
        self.assertFalse(practical_screen(row)['passed']);row['precision']=None
        self.assertFalse(practical_screen(row)['passed'])

    def test_prediction_freeze_has_no_outcomes_and_stress_keeps_primary_target(self):
        shape=(8,2,3);dates=pd.date_range('2025-01-01',periods=8,freq='6h');base=np.ones(np.prod(shape))
        block=dict(city='city',shape=shape,dates=dates,slice=slice(0,len(base)),seasonal=base,adaptive=base)
        predictions={m:(base,np.full(len(base),.2)) for m in METHODS if m not in ('seasonal','recent_calendar')}
        plan={'city':dict(dispersions={m:[.2]*3 for m in METHODS},calibrations={m:[dict(slope=1.,intercept=0.)]*3 for m in METHODS})}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cutoff=np.ones(shape,dtype=int)*9
            entries=freeze_forecasts({},[block],predictions,plan,root,'delayed_stress',{'city':cutoff})
            with np.load(root/entries[0]['file']) as saved:
                self.assertNotIn('counts',saved.files);self.assertNotIn('target',saved.files)
                self.assertEqual(saved['time_ns'][0],pd.Timestamp('2025-01-01').value)
                np.testing.assert_array_equal(saved['cutoff'],cutoff)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch not installed')
    def test_neural_ignores_held_out_training_targets_and_predicts_finite(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(120,4)).astype('float32');cat=np.zeros((120,3),dtype='int16')
        cat[:,0]=np.arange(120)%2;cat[:,2]=np.arange(120)%3
        base=np.ones(120)*2;y=rng.poisson(base);train=np.arange(120)<70;valid=(np.arange(120)>=70)&(np.arange(120)<100)
        first,_=fit_neural(x,cat,base,y,train,valid,7,epochs=1,local_city=0)
        changed=y.copy();changed[100:]+=1000;second,_=fit_neural(x,cat,base,changed,train,valid,7,epochs=1,local_city=0)
        for key in first['state_dict']:np.testing.assert_array_equal(first['state_dict'][key],second['state_dict'][key])
        mean,risk=predict_neural(x,cat,base,first);self.assertTrue(np.isfinite(mean).all());self.assertTrue(((risk>=0)&(risk<=1)).all())


if __name__=='__main__':unittest.main()
