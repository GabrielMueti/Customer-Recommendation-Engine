from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"
CSV_COLUMNS = ["customer_id", "customer_name", "rank", "product_id", "product_name", "category", "score", "supporting_customers"]


def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows). Decide what makes a purchase row unusable.

    Cleaning rules used here: IDs are trimmed and uppercased. Blank identity or
    catalog fields, nonpositive or noninteger quantities, and repeated purchase
    IDs are rejected (first occurrence wins).
    """
    valid_rows: list[dict] = []
    rejected_rows: list[dict] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"purchase_id", "customer_id", "customer_name", "product_id", "product_name", "category", "quantity"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for line_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items() if key is not None}
            for key in ("purchase_id", "customer_id", "product_id"):
                row[key] = row.get(key, "").upper()
            reason = ""
            if not row["purchase_id"]:
                reason = "blank purchase_id"
            elif row["purchase_id"] in seen_ids:
                reason = "duplicate purchase_id"
            else:
                seen_ids.add(row["purchase_id"])
                if not row["customer_id"] or not row.get("customer_name"):
                    reason = "missing customer ID or name"
                elif not row["product_id"] or not row.get("product_name") or not row.get("category"):
                    reason = "missing product ID, name, or category"
                else:
                    try:
                        quantity = int(row.get("quantity", ""))
                        if quantity <= 0:
                            reason = "quantity must be greater than zero"
                        else:
                            row["quantity"] = quantity
                    except ValueError:
                        reason = "quantity must be a whole number"
            row["source_row"] = line_number
            (rejected_rows if reason else valid_rows).append({**row, **({"reason": reason} if reason else {})})
    return valid_rows, rejected_rows


def jaccard(a: set, b: set) -> float:
    """|A intersection B| / |A union B|. Define (and test) what happens when both are empty.

    This implementation defines two empty sets as identical (1.0).
    """
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict], min_similarity: float = 0.0) -> None:
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        self.min_similarity = min_similarity
        histories: defaultdict[str, set[str]] = defaultdict(set)
        self.customer_names: dict[str, str] = {}
        self.product_names: dict[str, str] = {}
        self.product_categories: dict[str, str] = {}
        self.product_popularity: Counter[str] = Counter()
        for row in purchases:
            customer = str(row["customer_id"]).strip().upper()
            product = str(row["product_id"]).strip().upper()
            histories[customer].add(product)
            self.customer_names.setdefault(customer, str(row.get("customer_name", "")).strip())
            self.product_names.setdefault(product, str(row.get("product_name", "")).strip())
            self.product_categories.setdefault(product, str(row.get("category", "")).strip())
            self.product_popularity[product] += 1
        self.customer_products = dict(histories)

    def _customer(self, customer_id: str) -> str:
        customer = str(customer_id).strip().upper()
        if customer not in self.customer_products:
            raise KeyError(f"Unknown customer ID: {customer_id}")
        return customer

    def history(self, customer_id: str) -> set[str]:
        return set(self.customer_products[self._customer(customer_id)])

    def similarity(self, customer_a: str, customer_b: str) -> float:
        a = self._customer(customer_a)
        b =  self._customer(customer_b)
        return jaccard(self.customer_products[a], self.customer_products[b])

    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the k most similar other customers as (customer_id, score), best first."""
        customer = self._customer(customer_id)
        if k <= 0:
            return []
        neighbours = [(other, jaccard(self.customer_products[customer], items))
                      for other, items in self.customer_products.items() if other != customer]
        neighbours = [pair for pair in neighbours if pair[1] >= self.min_similarity]
        return sorted(neighbours, key=lambda pair: (-pair[1], pair[0]))[:k]

    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products bought by similar customers but not by this one, with the evidence for each."""
        customer = self._customer(customer_id)
        owned = self.customer_products[customer]
        neighbours = self.similar_customers(customer, k)
        candidates: dict[str, dict] = {}
        for neighbour, similarity in neighbours:
            for product in self.customer_products[neighbour] - owned:
                item = candidates.setdefault(product, {"neighbours": [], "weighted_support": 0.0})
                item["neighbours"].append({"customer_id": neighbour, "similarity": similarity})
                item["weighted_support"] += similarity
        fallback = not candidates
        maximum = max(self.product_popularity.values(), default=1)
        if fallback:
            for product, count in self.product_popularity.items():
                if product not in owned:
                    candidates[product] = {"neighbours": [], "weighted_support": 0.0, "fallback": True}
        for product, item in candidates.items():
            item.setdefault("fallback", False)
            item["product_id"] = product
            item["supporting_customers"] = sorted(n["customer_id"] for n in item["neighbours"])
            item["support_count"] = len(item["supporting_customers"])
            item["popularity_ratio"] = self.product_popularity[product] / maximum
        return candidates

    def score(self, evidence: dict) -> float:
        """Your ranking rule. Combine similarity with at least one other signal.

        Here the rule uses 70% similarity-weighted support and 30% popularity.
        Popularity alone is used for fallback candidates.
        """
        popularity = float(evidence.get("popularity_ratio", 0.0))
        if evidence.get("fallback"):
            return popularity
        total = float(evidence.get("total_neighbour_similarity", 0.0))
        support = float(evidence.get("weighted_support", 0.0))
        support_share = support / total if total else 0.0
        return 0.70 * support_share + 0.30 * popularity

    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Return up to n ranked recommendations for one customer."""
        customer = self._customer(customer_id)
        if n <= 0:
            return []
        candidates = self.candidate_products(customer)
        total_similarity = sum(value for _, value in self.similar_customers(customer))
        rows = []
        for product, evidence in candidates.items():
            scored = {**evidence, "total_neighbour_similarity": total_similarity}
            rows.append({
                "customer_id": customer, "customer_name": self.customer_names[customer],
                "product_id": product, "product_name": self.product_names[product],
                "category": self.product_categories[product], "score": self.score(scored),
                "supporting_customers": ";".join(evidence["supporting_customers"]),
                "support_count": evidence["support_count"],
            })
        rows.sort(key=lambda row: (-row["score"], -row["support_count"], -self.product_popularity[row["product_id"]], row["product_id"]))
        return [{**row, "rank": rank} for rank, row in enumerate(rows[:n], 1)]

    def recommend_all(self, n: int = 3) -> list[dict]:
        """Return recommendations for every known customer."""
        rows: list[dict] = []
        for customer in sorted(self.customer_products):
            recommendations = self.recommend(customer, n)
            if recommendations:
                rows.extend(recommendations)
            else:
                rows.append({"customer_id": customer, "customer_name": self.customer_names[customer],
                             "rank": "", "product_id": "", "product_name": "", "category": "",
                             "score": "", "supporting_customers": "", "support_count": 0})
        return rows

    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                output = {key: row.get(key, "") for key in CSV_COLUMNS}
                if output["score"] != "":
                    output["score"] = f"{float(output['score']):.3f}"
                writer.writerow(output)
