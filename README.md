# Mama Mboga Online — Customer Recommendation Engine

This project recommends grocery products a customer has not bought yet. It compares customers by the products in their purchase histories, using Jaccard similarity, then ranks products that similar customers bought. The approach uses Python's standard library and is simple to inspect and explain.

## Run the project

Use Python 3.10 or newer. No extra libraries or installation steps are needed. From the project folder, run:

```bash
python main.py
```

The program reads `data/purchases.csv`, prints a similarity matrix and sample recommendations, reports rejected rows, and writes `output/recommendations.csv`. It overwrites that output file each time it runs.

To run the optional checks in `tests.py`, use:

```bash
python -m unittest tests
```

## How the recommendations work

Each customer's history is a set of product IDs. Jaccard similarity measures the overlap between two customers' sets:

```text
similarity = number of shared products / number of distinct products between them
```

If both sets are empty, `jaccard()` returns `1.0`. The engine excludes a customer from their own neighbours. `min_similarity` defaults to `0.0`, so zero-similarity customers are eligible; set it above zero (for example, `0.01`) to exclude them. Neighbours are ordered by similarity, then customer ID, so ties are predictable.

For each customer, the engine finds the three most similar other customers by default. It gathers products those neighbours bought that the customer has not bought. Neighbours with higher similarity contribute more evidence.

The score combines similarity evidence with overall product popularity:

```text
score = 0.70 × weighted support share + 0.30 × popularity ratio
```

Weighted support share is the sum of the recommending neighbours' similarities divided by the sum of all selected neighbours' similarities. Popularity ratio is the product's number of accepted purchase rows divided by the largest such count for any product. When selected neighbours have no unseen products to suggest, the engine falls back to ranking unseen products by popularity. If selected neighbours have zero similarity but do suggest an unseen product, its support share is zero and its score comes from the 30% popularity component. Recommendation ties are resolved by support count, popularity, then product ID.

### Worked example

Two neighbours have similarities `0.50` and `0.25`, and both bought P03. Their total similarity is `0.75`, so P03's weighted support share is `0.75 / 0.75 = 1.0`. If P03 has 4 purchase rows and the most popular product has 8, its popularity ratio is `4 / 8 = 0.5`. The score is:

```text
0.70 × 1.0 + 0.30 × 0.5 = 0.850
```

## Data cleaning and assumptions

The loader trims whitespace and uppercases purchase, customer, and product IDs, so `c106` and `C106` are treated as the same customer. The first occurrence of each purchase ID is kept; later occurrences are rejected. Rows with a blank purchase ID, missing customer ID or name, missing product ID, name or category, or a quantity that is not a positive whole number are rejected with a source line number and reason. In particular, zero-quantity rows are rejected.

Purchase history is based on whether a customer bought an item, not how many units they bought, so each product appears once in a customer's set. Popularity counts accepted purchase rows rather than units. Reorders in separate purchase rows therefore contribute again to popularity. The first encountered customer name and product details for each normalized ID are used.

All customers with valid purchases appear in the CSV. If a customer has no unseen products to recommend, they receive one row with blank product fields. Unknown customer IDs passed to the engine raise a `KeyError`.

## Optional checks

`tests.py` has 11 standard-library `unittest` checks covering data cleaning, Jaccard edge cases, customer histories, neighbours, recommendations, the popularity fallback, unknown IDs, and CSV export. Run them with:

```bash
python -m unittest tests
```

The current dataset produces 20 customers, 15 products, 60 accepted purchase rows, and 3 rejected rows. The output CSV has the columns `customer_id,customer_name,rank,product_id,product_name,category,score,supporting_customers`.

## Project files

- `recommender.py` contains the CSV loader, Jaccard function, recommendation engine, and CSV export.
- `main.py` runs the workflow and prints the report.
- `data/purchases.csv` contains the input records.
- `output/recommendations.csv` contains the generated recommendations.
- `tests.py` contains the optional automated checks.
