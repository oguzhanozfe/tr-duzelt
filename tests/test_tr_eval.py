import json
import unittest
from io import BytesIO
from unittest.mock import patch

from tr_eval import EvaluationError, evaluate_correction


def gemini_response(text):
    return json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": text}],
            },
        }],
    }).encode("utf-8")


class EvaluateCorrectionTests(unittest.TestCase):
    def test_returns_exact_review_spans_and_sends_api_key_in_header(self):
        result_text = json.dumps({
            "issues": [{
                "original": "herkez ler",
                "corrected": "Herkes her",
                "reason": "The second word appears to be an incorrect substitution.",
                "rule_suggestion": "",
            }],
        })
        original = "herkez ler geldi mi"
        corrected = "Herkes her geldi mi"

        with patch("tr_eval.urlopen", return_value=BytesIO(gemini_response(result_text))) as request:
            result = evaluate_correction(original, corrected, "test-secret")

        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].original, "herkez ler")
        http_request = request.call_args.args[0]
        self.assertEqual(http_request.get_header("X-goog-api-key"), "test-secret")
        self.assertNotIn("test-secret", http_request.full_url)
        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(payload["contents"][0]["parts"][0]["text"],
                         json.dumps({"original": original, "corrected": corrected},
                                    ensure_ascii=False))

    def test_rejects_spans_not_present_in_input_and_output(self):
        result_text = json.dumps({
            "issues": [{
                "original": "not in input",
                "corrected": "not in output",
                "reason": "This span was not present.",
                "rule_suggestion": "",
            }],
        })
        with patch("tr_eval.urlopen", return_value=BytesIO(gemini_response(result_text))):
            with self.assertRaisesRegex(EvaluationError, "span that is not present"):
                evaluate_correction("girdi", "çıktı", "test-secret")

    def test_rejects_invalid_model_json(self):
        with patch("tr_eval.urlopen", return_value=BytesIO(gemini_response("not json"))):
            with self.assertRaisesRegex(EvaluationError, "not valid JSON"):
                evaluate_correction("girdi", "çıktı", "test-secret")

    def test_rejects_malformed_candidate(self):
        response = BytesIO(json.dumps({"candidates": ["unexpected"]}).encode("utf-8"))
        with patch("tr_eval.urlopen", return_value=response):
            with self.assertRaisesRegex(EvaluationError, "malformed candidate"):
                evaluate_correction("girdi", "çıktı", "test-secret")


if __name__ == "__main__":
    unittest.main()
