import os
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

DATASET_DIR = Path(__file__).parent / "sap-order-to-cash-dataset"
BATCH_SIZE = 1000

# Added the "items" folders which contain the relationship mappings
FOLDER_MAP = {
    "business_partners": "business_partners",
    "sales_order_headers": "sales_order_headers",
    "sales_order_items": "sales_order_items",
    "products": "products",
    "outbound_delivery_headers": "outbound_delivery_headers",
    "outbound_delivery_items": "outbound_delivery_items", 
    "billing_document_headers": "billing_document_headers",
    "billing_document_items": "billing_document_items",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def choose_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    cols_lower = {c: c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    raise ValueError(f"Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)}")

def safe_df(dataset_dir: Path, folder_name: str) -> pd.DataFrame:
    target_dir = None
    for path in dataset_dir.rglob(folder_name):
        if path.is_dir():
            target_dir = path
            break
            
    if not target_dir:
        raise FileNotFoundError(f"CRITICAL: Could not find folder '{folder_name}' inside {dataset_dir}")
        
    jsonl_files = list(target_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"CRITICAL: No .jsonl files found in {target_dir}")
        
    print(f"Found {len(jsonl_files)} file(s) for {folder_name}. Combining...")
    
    dfs = []
    for file in jsonl_files:
        df = pd.read_json(file, lines=True)
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = normalize_columns(combined_df)
    
    for col in combined_df.columns:
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            combined_df[col] = combined_df[col].fillna(0)
        else:
            combined_df[col] = combined_df[col].fillna("").astype(str)
            
    return combined_df

def chunked(records: List[Dict[str, Any]], size: int):
    for i in range(0, len(records), size):
        yield records[i : i + size]

def run_unwind_batches(driver, query: str, records: List[Dict[str, Any]], label: str) -> None:
    if not records:
        print(f"Skipped {label}: no records")
        return
    total = len(records)
    processed = 0
    with driver.session(database=USERNAME) as session:
        for batch in chunked(records, BATCH_SIZE):
            session.execute_write(lambda tx, b=batch: tx.run(query, rows=b).consume())
            processed += len(batch)
    print(f"Ingested {label}: {processed}/{total}")

def build_frames() -> Dict[str, pd.DataFrame]:
    return {key: safe_df(DATASET_DIR, folder_name) for key, folder_name in FOLDER_MAP.items()}

def prepare_records(frames: Dict[str, pd.DataFrame]):
    bp = frames["business_partners"]
    soh = frames["sales_order_headers"]
    soi = frames["sales_order_items"]
    prd = frames["products"]
    odh = frames["outbound_delivery_headers"]
    odi = frames["outbound_delivery_items"]
    bdh = frames["billing_document_headers"]
    bdi = frames["billing_document_items"]

    # Exact SAP column names added to candidates
    customer_id_col = choose_column(bp, ["businesspartner", "business_partner"], "Customer ID")
    order_id_col = choose_column(soh, ["salesorder", "sales_order"], "Order ID")
    order_customer_col = choose_column(soh, ["soldtoparty", "customer_id"], "Order -> Customer reference")
    
    item_order_col = choose_column(soi, ["salesorder", "sales_order"], "Order reference in items")
    item_product_col = choose_column(soi, ["product", "material"], "Product reference in items")
    product_id_col = choose_column(prd, ["product", "material"], "Product ID")
    
    delivery_id_col = choose_column(odh, ["deliverydocument", "outbounddelivery"], "Delivery ID")
    billing_id_col = choose_column(bdh, ["billingdocument", "billing_document"], "Billing Document ID")

    # The relationship links live in the Items files
    odi_del_col = choose_column(odi, ["deliverydocument", "outbounddelivery"], "Delivery ID in items")
    odi_ord_col = choose_column(odi, ["referencesddocument", "referencesalesorder", "salesorder"], "Delivery -> Order reference")

    bdi_bil_col = choose_column(bdi, ["billingdocument"], "Billing ID in items")
    bdi_del_col = choose_column(bdi, ["referencesddocument", "referencedocument"], "Billing -> Delivery reference")

    customers = [{"customer_id": str(row[customer_id_col]).strip(), "name": str(row.get("businesspartnerfullname", row.get("name", ""))).strip(), "country": str(row.get("country", "")).strip(), "city": str(row.get("city", "")).strip()} for _, row in bp.iterrows() if str(row[customer_id_col]).strip()]
    orders = [{"order_id": str(row[order_id_col]).strip(), "customer_id": str(row[order_customer_col]).strip(), "order_date": str(row.get("creationdate", "")).strip(), "currency": str(row.get("transactioncurrency", "")).strip()} for _, row in soh.iterrows() if str(row[order_id_col]).strip()]
    products = [{"product_id": str(row[product_id_col]).strip(), "product_name": str(row.get("productname", "")).strip(), "product_group": str(row.get("productgroup", "")).strip()} for _, row in prd.iterrows() if str(row[product_id_col]).strip()]
    deliveries = [{"delivery_id": str(row[delivery_id_col]).strip(), "order_id": "", "delivery_date": str(row.get("creationdate", "")).strip()} for _, row in odh.iterrows() if str(row[delivery_id_col]).strip()]
    billings = [{"billing_id": str(row[billing_id_col]).strip(), "delivery_id": "", "billing_date": str(row.get("billingdocumentdate", row.get("creationdate", ""))).strip()} for _, row in bdh.iterrows() if str(row[billing_id_col]).strip()]
    
    order_product_pairs = [{"order_id": str(row[item_order_col]).strip(), "product_id": str(row[item_product_col]).strip(), "quantity": float(row.get("requestedquantity", 0) or 0)} for _, row in soi.iterrows() if str(row[item_order_col]).strip() and str(row[item_product_col]).strip()]
    customer_order_pairs = [{"customer_id": o["customer_id"], "order_id": o["order_id"]} for o in orders if o["customer_id"] and o["order_id"]]
    order_delivery_pairs = [{"order_id": str(row[odi_ord_col]).strip(), "delivery_id": str(row[odi_del_col]).strip()} for _, row in odi.iterrows() if str(row[odi_ord_col]).strip() and str(row[odi_del_col]).strip()]
    delivery_billing_pairs = [{"delivery_id": str(row[bdi_del_col]).strip(), "billing_id": str(row[bdi_bil_col]).strip()} for _, row in bdi.iterrows() if str(row[bdi_del_col]).strip() and str(row[bdi_bil_col]).strip()]

    return {
        "customers": customers, "orders": orders, "products": products, "deliveries": deliveries,
        "billings": billings, "customer_order_pairs": customer_order_pairs, "order_product_pairs": order_product_pairs,
        "order_delivery_pairs": order_delivery_pairs, "delivery_billing_pairs": delivery_billing_pairs,
    }

def ingest():
    print(f"Reading data from: {DATASET_DIR}")
    frames = build_frames()
    data = prepare_records(frames)

    create_constraints = [
        "CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE",
        "CREATE CONSTRAINT order_id_unique IF NOT EXISTS FOR (o:Order) REQUIRE o.order_id IS UNIQUE",
        "CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
        "CREATE CONSTRAINT delivery_id_unique IF NOT EXISTS FOR (d:Delivery) REQUIRE d.delivery_id IS UNIQUE",
        "CREATE CONSTRAINT billing_id_unique IF NOT EXISTS FOR (b:BillingDocument) REQUIRE b.billing_id IS UNIQUE",
    ]

    q_customer = "UNWIND $rows AS row MERGE (c:Customer {customer_id: row.customer_id}) SET c.name = row.name, c.country = row.country, c.city = row.city"
    q_order = "UNWIND $rows AS row MERGE (o:Order {order_id: row.order_id}) SET o.order_date = row.order_date, o.currency = row.currency"
    q_product = "UNWIND $rows AS row MERGE (p:Product {product_id: row.product_id}) SET p.product_name = row.product_name, p.product_group = row.product_group"
    q_delivery = "UNWIND $rows AS row MERGE (d:Delivery {delivery_id: row.delivery_id}) SET d.delivery_date = row.delivery_date"
    q_billing = "UNWIND $rows AS row MERGE (b:BillingDocument {billing_id: row.billing_id}) SET b.billing_date = row.billing_date"
    q_customer_order = "UNWIND $rows AS row MATCH (c:Customer {customer_id: row.customer_id}) MATCH (o:Order {order_id: row.order_id}) MERGE (c)-[:PLACED]->(o)"
    q_order_product = "UNWIND $rows AS row MATCH (o:Order {order_id: row.order_id}) MATCH (p:Product {product_id: row.product_id}) MERGE (o)-[r:CONTAINS]->(p) SET r.quantity = row.quantity"
    q_order_delivery = "UNWIND $rows AS row MATCH (o:Order {order_id: row.order_id}) MATCH (d:Delivery {delivery_id: row.delivery_id}) MERGE (o)-[:FULFILLED_BY]->(d)"
    q_delivery_billing = "UNWIND $rows AS row MATCH (d:Delivery {delivery_id: row.delivery_id}) MATCH (b:BillingDocument {billing_id: row.billing_id}) MERGE (d)-[:BILLED_IN]->(b)"

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected to Neo4j.")

        with driver.session(database=USERNAME) as session:
            for q in create_constraints:
                session.run(q).consume()

        run_unwind_batches(driver, q_customer, data["customers"], "Customer nodes")
        run_unwind_batches(driver, q_order, data["orders"], "Order nodes")
        run_unwind_batches(driver, q_product, data["products"], "Product nodes")
        run_unwind_batches(driver, q_delivery, data["deliveries"], "Delivery nodes")
        run_unwind_batches(driver, q_billing, data["billings"], "BillingDocument nodes")

        run_unwind_batches(driver, q_customer_order, data["customer_order_pairs"], "PLACED relationships")
        run_unwind_batches(driver, q_order_product, data["order_product_pairs"], "CONTAINS relationships")
        run_unwind_batches(driver, q_order_delivery, data["order_delivery_pairs"], "FULFILLED_BY relationships")
        run_unwind_batches(driver, q_delivery_billing, data["delivery_billing_pairs"], "BILLED_IN relationships")

        print("Ingestion complete.")
    finally:
        driver.close()

if __name__ == "__main__":
    ingest()