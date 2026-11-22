from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from critical_processes.cad_forecast import cad_features


class CadContextTests(unittest.TestCase):
    def data(self):
        return dict(response_month=np.array([0,1,1,2]),closed_month=np.array([0,1,3,2]),areas=np.array([0,0,1,0]),
            counts=np.array([31,28,99,31]),channel_flags=np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,0,0]],dtype=bool))

    def test_nothing_before_first_month_publication(self):
        dates=pd.to_datetime(['2021-02-15','2021-02-16','2021-02-16 06:00'],format='mixed')
        result=cad_features(self.data(),dates)
        self.assertTrue((result[:2,:,:12]==0).all());self.assertAlmostEqual(float(result[2,0,0]),np.log(2),places=6)

    def test_late_closed_call_is_not_visible_early(self):
        dates=pd.to_datetime(['2021-03-17']);result=cad_features(self.data(),dates)
        self.assertEqual(result[0,1,3],0.)

    def test_future_values_do_not_change_earlier_features(self):
        dates=pd.date_range('2021-01-01','2021-03-20',freq='6h');first=self.data();second={k:v.copy() for k,v in first.items()};second['counts'][2:]=100000
        np.testing.assert_array_equal(cad_features(first,dates),cad_features(second,dates))

    def test_longer_publication_delay_never_exposes_recent_month(self):
        dates=pd.to_datetime(['2021-03-01']);early=cad_features(self.data(),dates,15);late=cad_features(self.data(),dates,45)
        self.assertGreater(early[0,0,0],late[0,0,0])

    def test_categories_are_initial_context_channels(self):
        result=cad_features(self.data(),pd.to_datetime(['2021-02-20']))
        self.assertEqual(result[0,0,0],result[0,0,1]);self.assertEqual(result[0,0,2],0.)

    def test_invalid_closure_order_rejected(self):
        data=self.data();data['closed_month'][0]=-1
        with self.assertRaises(ValueError):cad_features(data,pd.to_datetime(['2021-03-01']))


if __name__=='__main__':unittest.main()
