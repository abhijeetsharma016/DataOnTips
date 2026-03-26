# 🚀 DataOnTips — Graph RAG for SAP Order-to-Cash

🔗 **Live Demo:** [https://a7g978jyhfxnmwgpy3xbm8.streamlit.app/](https://a7g978jyhfxnmwgpy3xbm8.streamlit.app/)
## 📸 Demo Screenshot

![DataOnTips Screenshot](./assets/demo.png)

---

## 📌 Overview

**DataOnTips** is a **Graph RAG (Retrieval-Augmented Generation)** system built for SAP Order-to-Cash (O2C) data.

It transforms fragmented enterprise data (customers, orders, deliveries, billing documents) into a **connected graph**, enabling:

* 🔍 Visual exploration of business flows
* 💬 Natural language querying using LLMs
* 📊 Data-grounded business insights

Unlike traditional dashboards, this system allows **dynamic reasoning over relationships**, not just static queries.

---

## 🧠 Key Features

### 1. 🕸️ Graph-Based Data Modeling

* Converts SAP O2C dataset into a **Neo4j graph**
* Models real-world relationships:

  * `Customer → Order`
  * `Order → Product`
  * `Order → Delivery`
  * `Delivery → Billing`

👉 Built using batch ingestion with Cypher queries 

---

### 2. 📊 Interactive Graph Visualization

* Built using **Streamlit + streamlit-agraph**

* Displays:

  * Nodes (Customers, Orders, Products, etc.)
  * Relationships (PLACED, CONTAINS, FULFILLED_BY, BILLED_IN)

* Supports:

  * Node highlighting
  * Relationship exploration
  * Real-time graph preview

---

### 3. 💬 Conversational Query Interface (Graph RAG)

* Powered by:

  * **LangChain**
  * **Google Gemini (gemini-2.5-flash)**

* Converts natural language → Cypher → executes on Neo4j → returns grounded answers

Example:

```
Q: Trace the flow for billing document 90504270
A: Sales Order → Delivery → Billing flow
```

👉 Implemented via `GraphCypherQAChain` 

---

### 4. 🧩 Custom Business Logic Layer

Handles domain-specific queries **without LLM hallucination**:

* 📈 Top products by billing volume
* 🔄 End-to-end flow tracing
* ⚠️ Anomaly detection (Delivered but not billed)

---

### 5. 🛡️ Guardrails (Important)

* Rejects irrelevant queries:

```
"This system is designed to answer questions related to the provided dataset only."
```

* Domain filtering via keyword detection
* Strict schema-bound Cypher generation

---

## 🏗️ Architecture

```
                ┌────────────────────────┐
                │   SAP O2C Dataset      │
                └──────────┬─────────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │  Data Ingestion Layer  │
                │ (Pandas + Neo4j)       │
                └──────────┬─────────────┘
                           ▼
                ┌────────────────────────┐
                │     Neo4j Graph DB     │
                └──────────┬─────────────┘
                           ▼
        ┌───────────────────────────────┐
        │   LangChain Graph RAG Layer   │
        │ (Cypher Generation + QA)      │
        └──────────┬────────────────────┘
                   ▼
        ┌───────────────────────────────┐
        │      Streamlit Frontend       │
        │  - Graph Visualization        │
        │  - Chat Interface             │
        └───────────────────────────────┘
```

---

## 🗄️ Graph Schema

### Nodes:

* `Customer`
* `Order`
* `Product`
* `Delivery`
* `BillingDocument`

### Relationships:

* `(:Customer)-[:PLACED]->(:Order)`
* `(:Order)-[:CONTAINS]->(:Product)`
* `(:Order)-[:FULFILLED_BY]->(:Delivery)`
* `(:Delivery)-[:BILLED_IN]->(:BillingDocument)`

---

## ⚙️ Tech Stack

| Layer           | Technology       |
| --------------- | ---------------- |
| Frontend        | Streamlit        |
| Graph DB        | Neo4j            |
| LLM             | Google Gemini    |
| Framework       | LangChain        |
| Data Processing | Pandas           |
| Visualization   | streamlit-agraph |

---

## 🚀 Setup Instructions

### 1. Clone Repo

```bash
git clone https://github.com/abhijeetsharma016/DataOnTips.git
cd DataOnTips
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup Environment Variables

Create a `.env` file:

```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
GOOGLE_API_KEY=your_google_api_key
```

---

### 4. Ingest Data into Neo4j

```bash
python ingest.py
```

👉 Builds full graph with relationships 

---

### 5. Run App

```bash
streamlit run app.py
```

---

## 💡 Example Queries

* 🔍 *Which products have the highest billing count?*
* 🔄 *Trace flow for billing document 90504270*
* ⚠️ *Find orders delivered but not billed*
* 👤 *What city is customer 320000083 in?*

---

## 🧠 LLM Prompting Strategy

### Cypher Generation Prompt

* Strict schema adherence
* No hallucinated fields
* Enforces string IDs

### QA Prompt

* Only uses DB context
* No fabricated answers
* Fallback for missing data

---

## 📈 Key Design Decisions

### ✅ Why Graph Database?

* O2C data is inherently relational
* Graph traversal is natural for:

  * Flow tracing
  * Dependency analysis

### ✅ Why Graph RAG?

* Combines:

  * Structured querying (Cypher)
  * Natural language flexibility (LLM)

### ✅ Why Custom Query Layer?

* Prevents LLM overuse
* Ensures accuracy for critical queries

---

## ⚡ Performance Optimizations

* Batch ingestion (`UNWIND`)
* Indexed constraints
* Query result limits
* Cached graph schema

---

## 🔮 Future Improvements

* Highlight nodes involved in answers
* Add semantic + hybrid search
* Conversation memory
* Graph clustering & analytics
* Streaming responses

---

## 📜 Assignment Context

This project was built as part of a **Graph-Based Data Modeling and Query System** challenge, focusing on:

* Graph modeling
* LLM-based querying
* Real-world business reasoning

👉 Full problem description 

---

## 👨‍💻 Author

**Abhijeet Sharma**

* 💻 Android Developer | DSA Enthusiast
* 🔗 GitHub: [https://github.com/abhijeetsharma016](https://github.com/abhijeetsharma016)

---

## ⭐ If you like this project

Give it a star ⭐ — it helps a lot!
