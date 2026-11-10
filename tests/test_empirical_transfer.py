import unittest
import numpy as np
import pandas as pd
from critical_processes.empirical_transfer import block_intervals,prepare_block


class EmpiricalTransferTests(unittest.TestCase):
    def test_block_selection_is_calendar_based_and_never_exceeds_available_period(self):
        portal=dict(source='portal:chicago',start='2019-01-01',end_exclusive='2026-10-01')
        blocks=block_intervals(portal)
        self.assertEqual([r[0] for r in blocks],['early','latest'])
        for name,start,end in blocks:self.assertEqual((end-start).days,1810)
        self.assertEqual(blocks[-1][2],pd.Timestamp('2026-10-01'))
        archive=block_intervals(dict(source='CODE:Boston',start='2018-01-01',end_exclusive='2023-01-01'))
        self.assertEqual(len(archive),1)
        self.assertEqual(archive[0][1],pd.Timestamp('2018-01-01'))

    def test_event_bounds_and_tied_timestamps_are_preserved(self):
        times=pd.to_datetime(['2019-12-31 23:59:00','2020-01-01 00:00:00','2020-01-01 00:00:00',
            '2020-01-02 23:59:00','2020-01-03 00:00:00']).as_unit('ns').astype('int64').to_numpy()
        events,counts=prepare_block(times,'2020-01-01','2020-01-03')
        np.testing.assert_array_equal(counts,[2,1])
        self.assertEqual(events[0],events[1])
        self.assertEqual(len(events),3)


if __name__=='__main__':unittest.main()
