import unittest

from app.services import prompt_examples_service


class PromptExamplesServiceTests(unittest.TestCase):
    def test_default_pool_can_fill_home_starter_row(self):
        self.assertGreaterEqual(len(prompt_examples_service.DEFAULT_EXAMPLES), 9)

    def test_balanced_sample_does_not_mutate_source_pool(self):
        examples = [
            {"category": "news", "question": "新闻问题一"},
            {"category": "news", "question": "新闻问题二"},
            {"category": "tech", "question": "技术问题一"},
            {"category": "tech", "question": "技术问题二"},
            {"category": "general", "question": "通用问题一"},
            {"category": "general", "question": "通用问题二"},
        ]
        original = [dict(item) for item in examples]

        sampled = prompt_examples_service._balanced_sample(examples, 3)

        self.assertEqual(len(sampled), 3)
        self.assertEqual(examples, original)
        self.assertEqual({item["category"] for item in sampled}, {"news", "tech", "general"})


if __name__ == "__main__":
    unittest.main()
