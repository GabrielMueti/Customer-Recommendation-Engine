from recommender import DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases


def print_similarity_matrix(engine: RecommendationEngine, limit: int = 5) -> None:
    customers = sorted(engine.customer_products)[:limit]
    print("Jaccard similarity matrix (first five customers):")
    print("customer," + ",".join(customers))
    for customer in customers:
        values = [f"{engine.similarity(customer, other):.2f}" for other in customers]
        print(f"{customer}," + ",".join(values))


def print_customer_report(engine: RecommendationEngine, customer_id: str, n: int = 3) -> None:
    history = ", ".join(sorted(engine.history(customer_id)))
    print(f"\nCustomer {customer_id} ({engine.customer_names[customer_id]}) bought: {{{history}}}")
    neighbours = engine.similar_customers(customer_id)
    print("Most similar customers:")
    if not neighbours:
        print("  none; using popularity fallback")
    for other, similarity in neighbours:
        print(f"  {other}  jaccard={similarity:.2f}")
    print("Recommendations:")
    recommendations = engine.recommend(customer_id, n)
    if not recommendations:
        print("  none available (customer has bought every known product)")
    for row in recommendations:
        why = f"{row['support_count']} similar customer(s) bought it" if row["support_count"] else "popular item fallback"
        print(f"  {row['rank']}. {row['product_name']} ({row['category']})  score={row['score']:.3f}  because {why}")


def main() -> None:
    purchases, rejected = read_purchases(DATA_PATH)
    engine = RecommendationEngine(purchases)
    print_similarity_matrix(engine)
    for customer in sorted(engine.customer_products)[:2]:
        print_customer_report(engine, customer)
    rows = engine.recommend_all(n=3)
    engine.export_csv(rows, OUTPUT_PATH)
    print("\nRejected rows:")
    for row in rejected:
        print(f"  source row {row['source_row']}: {row['reason']} (purchase {row.get('purchase_id', '')})")
    written = sum(bool(row["product_id"]) for row in rows)
    print("\nRun summary")
    print(f"Purchases read: {len(purchases) + len(rejected)} | used: {len(purchases)} | rejected: {len(rejected)}")
    print(f"Customers: {len(engine.customer_products)} | Products: {len(engine.product_names)} | Recommendations written: {written}")
    print(f"CSV: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
