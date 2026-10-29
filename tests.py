
"""Small regression checks for the recommendation engine. Run: python -m unittest tests"""

import csv
import tempfile
import unittest
from pathlib import Path

from recommender import RecommendationEngine, jaccard, read_purchases


def purchase(customer, product, *, name=None, product_name=None, quantity=1, purchase_id=None):
    return {
        "purchase_id": purchase_id or f"T-{customer}-{product}",
        "customer_id": customer,
        "customer_name": name or customer,
        "product_id": product,
        "product_name": product_name or f"Product {product}",
        "category": "Grocery",
        "quantity": quantity,
    }


class JaccardTests(unittest.TestCase):
    def test_overlap_is_intersection_over_union(self):
        self.assertAlmostEqual(jaccard({"P1", "P2"}, {"P2", "P3"}), 1 / 3)

    def test_two_empty_sets_have_similarity_one(self):
        self.assertEqual(jaccard(set(), set()), 1.0)

    def test_disjoint_sets_have_similarity_zero(self):
        self.assertEqual(jaccard({"P1"}, {"P2"}), 0.0)


class PurchaseLoadingTests(unittest.TestCase):
    def test_normalizes_ids_and_rejects_duplicate_missing_product_and_zero_quantity(self):
        content = (
            "purchase_id,customer_id,customer_name,product_id,product_name,category,quantity\n"
            "T1,c106,Mwangi,P1,Milk,Dairy,2\n"
            "T1,C106,Mwangi,P1,Milk,Dairy,2\n"
            "T2,C105,Njeri,,,Fresh,1\n"
            "T3,C104,Kamau,P2,Rice,Pantry,0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "purchases.csv"
            path.write_text(content, encoding="utf-8")
            valid, rejected = read_purchases(path)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["customer_id"], "C106")
        self.assertEqual(valid[0]["quantity"], 2)
        self.assertEqual([row["reason"] for row in rejected], [
            "duplicate purchase_id", "missing product ID, name, or category",
            "quantity must be greater than zero",
        ])


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            purchase("C1", "P1"), purchase("C1", "P2"),
            purchase("C2", "P1"), purchase("C2", "P2"), purchase("C2", "P3"),
            purchase("C3", "P1"), purchase("C3", "P3"),
        ]
        self.engine = RecommendationEngine(self.rows)

    def test_history_is_a_set_and_returned_as_a_copy(self):
        history = self.engine.history("c1")
        self.assertEqual(history, {"P1", "P2"})
        history.add("P99")
        self.assertNotIn("P99", self.engine.history("C1"))

    def test_similarity_and_neighbours(self):
        self.assertAlmostEqual(self.engine.similarity("C1", "C2"), 2 / 3)
        neighbours = self.engine.similar_customers("C1", k=2)
        self.assertEqual([customer for customer, _ in neighbours], ["C2", "C3"])
        self.assertNotIn("C1", [customer for customer, _ in neighbours])

    def test_recommendations_only_include_unowned_products(self):
        recommendations = self.engine.recommend("C1", n=3)
        self.assertEqual([row["product_id"] for row in recommendations], ["P3"])
        self.assertEqual(recommendations[0]["supporting_customers"], "C2;C3")

    def test_unknown_customer_raises_key_error(self):
        with self.assertRaisesRegex(KeyError, "Unknown customer ID"):
            self.engine.recommend("C999")

    def test_popularity_fallback_when_neighbours_have_no_unseen_items(self):
        rows = [purchase("C1", "P1"), purchase("C2", "P1"), purchase("C3", "P2")]
        # Exclude the zero-similarity customer so the fallback is exercised.
        engine = RecommendationEngine(rows, min_similarity=0.01)
        recs = engine.recommend("C1", n=1)
        self.assertEqual(recs[0]["product_id"], "P2")
        self.assertEqual(recs[0]["support_count"], 0)

    def test_every_customer_is_exported_even_without_a_recommendation(self):
        rows = [purchase("C1", "P1"), purchase("C1", "P2"), purchase("C2", "P1")]
        engine = RecommendationEngine(rows)
        all_rows = engine.recommend_all(n=3)
        self.assertEqual({row["customer_id"] for row in all_rows}, {"C1", "C2"})
        no_recommendation = [row for row in all_rows if row["customer_id"] == "C1"]
        self.assertEqual(len(no_recommendation), 1)
        self.assertEqual(no_recommendation[0]["product_id"], "")

    def test_csv_export_uses_expected_header_and_three_decimal_scores(self):
        rows = self.engine.recommend_all(n=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recommendations.csv"
            self.engine.export_csv(rows, path)
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                exported = list(reader)
                self.assertEqual(reader.fieldnames, [
                    "customer_id", "customer_name", "rank", "product_id",
                    "product_name", "category", "score", "supporting_customers",
                ])
        self.assertTrue(all(not row["score"] or len(row["score"].split(".")[-1]) == 3 for row in exported))


if __name__ == "__main__":
    unittest.main()
