from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from critical_processes.adaptive_alarm import calibration_masks,fit_policy,apply_policy,alarm_metrics,choose_candidate,preliminary_screen
from critical_processes.adaptive_alarm_experiment import append_frame
from critical_processes.offline_forecast import CategoryPanel


class AdaptiveAlarmTests(unittest.TestCase):
    def test_probability_threshold_and_maturity_blocks_are_disjoint_and_past(self):
        dates=pd.date_range('2024-01-01','2026-01-01',freq='6h',inclusive='left')
        masks=calibration_masks(dates,'2025-01-01',180)
        self.assertFalse(np.any(masks['probability']&masks['threshold']))
        self.assertLess(dates[masks['threshold']][-1],pd.Timestamp('2024-12-04'))

    def test_future_labels_and_scores_cannot_change_an_earlier_policy(self):
        dates=pd.date_range('2024-01-01','2026-01-01',freq='6h',inclusive='left');rng=np.random.default_rng(503)
        raw=rng.uniform(.01,.8,size=(len(dates),2,3));cutoff=np.ones_like(raw)*3;target=rng.poisson(1.4,size=raw.shape)
        first=fit_policy(raw,cutoff,target,dates,'2025-01-01',180,.05)
        future=dates>=pd.Timestamp('2024-12-04');raw[future]=.999;target[future]+=1000
        second=fit_policy(raw,cutoff,target,dates,'2025-01-01',180,.05)
        self.assertEqual(first,second)

    def test_tied_scores_conservatively_abstain_and_have_no_fake_precision(self):
        dates=pd.date_range('2024-01-01','2025-01-01',freq='6h',inclusive='left');raw=np.ones((len(dates),2,3))*.2
        policy=fit_policy(raw,np.ones_like(raw)*3,np.zeros_like(raw),dates,'2025-01-01',90,.01)
        probability,alarm=apply_policy(raw[-4:],policy);self.assertFalse(alarm.any())
        metric=alarm_metrics(np.zeros((4,2)),np.ones((4,2))*3,probability[:,:,0],alarm[:,:,0])
        self.assertIsNone(metric['precision']);self.assertEqual(metric['fbeta_half'],0.)

    def test_fbeta_and_false_alarm_denominators(self):
        y=np.array([[4,0],[0,4]]);p=np.array([[.9,.8],[.1,.2]]);alarm=p>=.5
        metric=alarm_metrics(y,np.ones_like(y)*3,p,alarm)
        self.assertEqual(metric['precision'],.5);self.assertEqual(metric['recall'],.5);self.assertEqual(metric['fpr'],.5)
        self.assertEqual(metric['fbeta_half'],.5)

    def test_joint_selection_prefers_safe_feasible_point_over_large_ap_or_fbeta(self):
        def row(name,precision,recall,fpr,fbeta):
            part=dict(precision=precision,recall=recall,fpr=fpr)
            return dict(id=name,method='local_neural',window_days=90,alpha=.05,development2025=part,development2026=part,combined=dict(fbeta_half=fbeta))
        choice=choose_candidate([row('unsafe',.9,.9,.2,.9),row('safe',.31,.12,.04,.24),row('too_rare',.8,.01,.001,.05)])
        self.assertEqual(choice['candidate'],'safe');self.assertEqual(choice['selection_status'],'fpr_and_recall_feasible')

    def test_no_safe_candidate_fallback_minimizes_fpr(self):
        rows=[]
        for name,fpr,fbeta in [('low',.06,.1),('high',.2,.9)]:
            part=dict(precision=.3,recall=.2,fpr=fpr)
            rows.append(dict(id=name,method='seasonal',window_days=90,alpha=.05,development2025=part,development2026=part,combined=dict(fbeta_half=fbeta)))
        self.assertEqual(choose_candidate(rows)['candidate'],'low')

    def test_short_new_test_never_proves_full_utility(self):
        row=dict(precision=.5,recall=.3,fpr=.01,ap=.5,baseline_ap=.2,brier=.1,baseline_brier=.2,blocks=3,brier_difference_ci_high=None)
        result=preliminary_screen(row);self.assertTrue(result['point_with_baseline_improvement_passed']);self.assertFalse(result['full_screen_passed'])
        self.assertEqual(result['statistical_status'],'preliminary_insufficient_time_blocks')

    def test_october_extension_preserves_history_and_deduplicates_cases(self):
        dates=pd.date_range('2019-01-01','2026-10-01',freq='6h',inclusive='left').as_unit('ns');counts=np.zeros((len(dates),2,3),dtype='int32')
        counts[0,0,[0,1]]=1
        old=CategoryPanel(counts,dates.asi8,np.array(['EAST','WEST']),np.array(['all','property','assault_robbery']),np.array([[47.,-122.],[47.1,-122.1]]),np.array([0]),np.array([0]),np.array([1]),np.array([0]))
        rows=pd.DataFrame([
            ['old','2026-10-01','2026-10-02','East','ROBBERY'],
            ['a','2026-10-02','2026-10-03','East','BURGLARY'],
            ['a','2026-10-02','2026-10-03','East','ROBBERY'],
            ['b','2026-10-03','2026-10-04','West','BURGLARY']],
            columns=['report_number','offense_date','report_date_time','precinct','offense_sub_category'])
        extended,quality=append_frame(old,rows,{'old'})
        np.testing.assert_array_equal(extended.counts[:len(old.counts)],old.counts)
        self.assertEqual(quality['new_cases'],2);self.assertEqual(quality['excluded_existing_history_rows'],1)
        self.assertEqual(extended.counts[len(old.counts):,:,0].sum(),2)
        self.assertEqual(extended.counts[len(old.counts):,:,2].sum(),1)


if __name__=='__main__':unittest.main()
