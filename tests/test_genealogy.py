import unittest
import numpy as np
from critical_processes.genealogy import sample_clusters, simulate_genealogical_cluster


class GenealogyTests(unittest.TestCase):
    def test_zero_branching_is_one_root_and_zero_duration(self):
        sizes,duration,generation=sample_clusters(0.,replicates=100,seed=2)
        np.testing.assert_array_equal(sizes,np.ones(100))
        self.assertTrue((duration==0).all() and (generation==0).all())

    def test_each_descendant_has_a_past_parent_and_generation(self):
        times,parents,depth=simulate_genealogical_cluster(.97,seed=491)
        self.assertEqual(parents[0],-1)
        for index in range(1,len(times)):
            self.assertLess(parents[index],index)
            self.assertGreater(times[index],times[parents[index]])
            self.assertEqual(depth[index],depth[parents[index]]+1)

    def test_complete_sizes_recover_branching_moments(self):
        sizes,_,_=sample_clusters(.5,replicates=100000,seed=81)
        self.assertAlmostEqual(sizes.mean(),2.,delta=.025)
        self.assertAlmostEqual(sizes.var(),4.,delta=.2)

    def test_decay_changes_chronology_but_not_tree_size_or_generation(self):
        first=simulate_genealogical_cluster(.9,decay=1.,seed=123)
        second=simulate_genealogical_cluster(.9,decay=2.,seed=123)
        np.testing.assert_allclose(first[0],2*second[0])
        np.testing.assert_array_equal(first[1],second[1])
        np.testing.assert_array_equal(first[2],second[2])

    def test_event_limit_raises_without_returning_a_biased_truncated_cluster(self):
        with self.assertRaises(RuntimeError):
            sample_clusters(.97,replicates=1000,seed=4,max_events=1)


if __name__=="__main__": unittest.main()
