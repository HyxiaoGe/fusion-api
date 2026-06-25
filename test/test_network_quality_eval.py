import io
import json
import tempfile
import unittest
from pathlib import Path


class NetworkQualityEvalTests(unittest.TestCase):
    def test_load_samples_rejects_duplicate_ids(self):
        from scripts.network_quality_eval import load_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "same",
                            "category": "news",
                            "question": "问题 1",
                            "query": "query 1",
                            "intent": "freshness",
                            "expected_domains": ["example.com"],
                        },
                        {
                            "id": "same",
                            "category": "news",
                            "question": "问题 2",
                            "query": "query 2",
                            "intent": "freshness",
                            "expected_domains": ["example.org"],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "重复样本 id"):
                load_samples(path)

    def test_score_results_rewards_expected_and_official_sources(self):
        from scripts.network_quality_eval import score_results

        sample = {
            "id": "official-doc",
            "expected_domains": ["openai.com"],
            "official_domains": ["platform.openai.com"],
            "min_results": 3,
        }
        results = [
            {"title": "Docs", "url": "https://platform.openai.com/docs/guides/tools"},
            {"title": "Pricing", "url": "https://openai.com/api/pricing"},
            {"title": "Blog", "url": "https://example.com/post"},
        ]

        score = score_results(sample, results)

        self.assertGreaterEqual(score["score"], 80)
        self.assertEqual(score["result_count"], 3)
        self.assertEqual(score["expected_domain_hits"], ["openai.com"])
        self.assertEqual(score["official_domain_hits"], ["platform.openai.com"])

    def test_score_results_penalizes_duplicate_domains(self):
        from scripts.network_quality_eval import score_results

        sample = {
            "id": "comparison",
            "expected_domains": ["apple.com", "theverge.com"],
            "official_domains": ["apple.com"],
            "min_results": 4,
        }
        results = [
            {"title": "A", "url": "https://reddit.com/r/apple/a"},
            {"title": "B", "url": "https://reddit.com/r/apple/b"},
            {"title": "C", "url": "https://reddit.com/r/apple/c"},
            {"title": "D", "url": "https://reddit.com/r/apple/d"},
        ]

        score = score_results(sample, results)

        self.assertLess(score["score"], 50)
        self.assertEqual(score["duplicate_domain_count"], 3)
        self.assertEqual(score["expected_domain_hits"], [])

    def test_score_results_does_not_treat_parent_domain_as_specific_subdomain_hit(self):
        from scripts.network_quality_eval import score_results

        sample = {
            "id": "specific-docs",
            "expected_domains": ["platform.openai.com"],
            "official_domains": ["platform.openai.com"],
            "min_results": 1,
        }
        results = [{"title": "OpenAI", "url": "https://openai.com/"}]

        score = score_results(sample, results)

        self.assertEqual(score["expected_domain_hits"], [])
        self.assertEqual(score["official_domain_hits"], [])

    def test_score_results_counts_most_specific_expected_domain_once(self):
        from scripts.network_quality_eval import score_results

        sample = {
            "id": "overlapping-domains",
            "expected_domains": ["openai.com", "platform.openai.com"],
            "official_domains": ["openai.com", "platform.openai.com"],
            "min_results": 3,
        }
        results = [
            {"title": "Docs", "url": "https://platform.openai.com/docs"},
            {"title": "Other 1", "url": "https://example.com/a"},
            {"title": "Other 2", "url": "https://example.org/b"},
        ]

        score = score_results(sample, results)

        self.assertEqual(score["expected_domain_hits"], ["platform.openai.com"])
        self.assertEqual(score["official_domain_hits"], ["platform.openai.com"])
        self.assertLess(score["score"], 100)

    def test_write_dry_run_outputs_jsonl(self):
        from scripts.network_quality_eval import write_dry_run

        samples = [
            {
                "id": "latest-price",
                "category": "price",
                "question": "最新价格是什么？",
                "query": "OpenAI API pricing latest",
                "intent": "freshness",
                "expected_domains": ["openai.com"],
                "official_domains": ["openai.com"],
                "min_results": 3,
            }
        ]
        output = io.StringIO()

        write_dry_run(samples, output)

        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["sample_id"], "latest-price")
        self.assertEqual(lines[0]["query"], "OpenAI API pricing latest")
        self.assertEqual(lines[0]["expected_domains"], ["openai.com"])
        self.assertEqual(lines[0]["score"], 0)

    def test_default_sample_file_is_valid(self):
        from scripts.network_quality_eval import DEFAULT_SAMPLE_PATH, load_samples

        samples = load_samples(DEFAULT_SAMPLE_PATH)

        self.assertGreaterEqual(len(samples), 5)
        self.assertEqual({sample["id"] for sample in samples}, set(sample["id"] for sample in samples))


if __name__ == "__main__":
    unittest.main()
