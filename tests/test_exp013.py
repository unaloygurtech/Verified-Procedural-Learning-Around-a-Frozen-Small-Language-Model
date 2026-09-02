import unittest

from air_core.exp013 import (
    CapabilityFingerprint,
    CapabilityQuery,
    HierarchicalCapabilityIndex,
    benchmark_retrieval_strategy,
    deduplicate_skills,
    generate_capability_records,
    make_composition_library,
    make_dedup_fixture,
    run_composition_block,
    run_context_block,
    run_dedup_block,
    run_utilization_block,
)


class Experiment013Tests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_round_trips(self) -> None:
        item = generate_capability_records(3)[1]
        payload = item.fingerprint.encode()
        self.assertEqual(payload, item.fingerprint.encode())
        self.assertEqual(CapabilityFingerprint.decode(payload), item.fingerprint)
        self.assertLess(len(payload), 180)

    def test_one_canonical_artifact_has_many_facets(self) -> None:
        item = generate_capability_records(1)[0]
        index = HierarchicalCapabilityIndex([item])
        self.assertGreaterEqual(len(item.facets), 5)
        self.assertEqual(index.canonical_artifact_bytes(), len(item.record.artifact))
        self.assertTrue(all(item.skill_id in ids for ids in index.by_facet.values()))

    def test_partial_three_of_five_facets_does_not_overfilter(self) -> None:
        item = generate_capability_records(100)[42]
        index = HierarchicalCapabilityIndex(generate_capability_records(100))
        query = CapabilityQuery(
            input_type=item.fingerprint.input_type,
            output_type=item.fingerprint.output_type,
            operation_family=item.fingerprint.operation_family,
            facets=item.facets[:3],
        )
        found, telemetry = index.retrieve(query, top_k=5)
        self.assertIn(item.skill_id, {candidate.skill_id for candidate in found})
        self.assertLess(telemetry["candidates_examined"], 100)
        union_found, union_telemetry = index.retrieve_union(
            CapabilityQuery(facets=item.facets[:3], trust="verified"), top_k=5
        )
        self.assertIn(item.skill_id, {candidate.skill_id for candidate in union_found})
        self.assertGreaterEqual(union_telemetry["candidates_examined"], telemetry["candidates_examined"])

    def test_fingerprint_filter_preserves_accuracy(self) -> None:
        records = generate_capability_records(300)
        target = records[123]
        query = CapabilityQuery(
            input_type=target.fingerprint.input_type,
            output_type=target.fingerprint.output_type,
            operation_family=target.fingerprint.operation_family,
            trust="verified",
        )
        no_fp = benchmark_retrieval_strategy(records, query, strategy="hierarchical_no_fingerprint", target_skill_id=target.skill_id)
        with_fp = benchmark_retrieval_strategy(records, query, strategy="hierarchical_fingerprint", target_skill_id=target.skill_id)
        self.assertEqual(no_fp["top5_recall"], with_fp["top5_recall"])
        self.assertLessEqual(with_fp["candidates_examined"], no_fp["candidates_examined"])

    def test_unknown_domain_is_safe(self) -> None:
        index = HierarchicalCapabilityIndex(generate_capability_records(100))
        found, telemetry = index.retrieve(CapabilityQuery(domain="does-not-exist"), top_k=5)
        self.assertEqual(found, [])
        self.assertEqual(telemetry["candidates_examined"], 0)

    def test_dedup_merges_equivalent_but_not_edge_difference(self) -> None:
        result = deduplicate_skills(make_dedup_fixture())
        self.assertEqual(result.false_merges, 0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertLess(len(result.representatives), 6)
        self.assertTrue(any(set(group) == {"trim-lower", "strip-lower", "different-provenance"} for group in result.groups))
        self.assertTrue(all(item.provenance for item in result.representatives))

    def test_dedup_report_contains_storage_saving(self) -> None:
        report = run_dedup_block()
        self.assertGreater(report["storage_bytes_saved"], 0)
        self.assertEqual(report["false_merges"], 0)

    def test_composition_scoping_preserves_valid_and_rejects_missing(self) -> None:
        report = run_composition_block()
        self.assertTrue(report["all_accuracy_preserved"])
        self.assertGreater(report["agent_candidate_reduction_vs_global"], 0.99)
        missing = next(row for row in report["tasks"] if row["task"] == "missing-capability")
        self.assertTrue(missing["no_valid_composition_correct"])
        self.assertLessEqual(report["bounded_subagents"], 3)

    def test_context_compression_does_not_reduce_quality(self) -> None:
        report = run_context_block()
        rows = {row["condition"]: row for row in report["conditions"]}
        self.assertTrue(all(row["downstream_correct"] for row in rows.values()))
        self.assertGreater(rows["top_k_relevant_snippet"]["context_token_reduction_vs_full"], 0.9)
        self.assertFalse(report["hard_budget_default"])

    def test_utilization_workload_has_safe_unknowns(self) -> None:
        report = run_utilization_block(generate_capability_records(1_000))
        self.assertEqual(report["safe_unknown_rate"], 1.0)
        self.assertTrue(any(row["layers_visited"] for row in report["workloads"]))
        self.assertTrue(any(row["workload"] == "exact_known" and row["correct_skill"] for row in report["workloads"]))


if __name__ == "__main__":
    unittest.main()
