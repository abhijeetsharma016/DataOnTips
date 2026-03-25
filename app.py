import os
import re
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from streamlit_agraph import Config, Edge, Node, agraph

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as exc:
    raise ImportError(
        "Missing dependency: langchain-google-genai. Install with `pip install langchain-google-genai`."
    ) from exc

try:
    from langchain.chains import GraphCypherQAChain
except ImportError:
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain


load_dotenv()

APP_TITLE = "DataOnTips - Graph RAG for SAP O2C"
DEFAULT_REJECTION = "This system is designed to answer questions related to the provided dataset only."
MAX_GRAPH_RELATIONS = 50

NODE_COLORS = {
    "Customer": "#4F9DDE",
    "Order": "#F6C85F",
    "Product": "#6FB07F",
    "Delivery": "#F08A5D",
    "BillingDocument": "#B39DDB",
}

DOMAIN_KEYWORDS = {
    "order",
    "orders",
    "sales",
    "delivery",
    "deliveries",
    "billing",
    "invoice",
    "customer",
    "customers",
    "product",
    "products",
    "sap",
    "o2c",
    "order-to-cash",
    "flow",
    "billed",
    "unbilled",
    "dataset",
    "neo4j",
    "document",
}


def get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@st.cache_resource(show_spinner=False)
def init_services():
    uri = get_env("NEO4J_URI")
    username = get_env("NEO4J_USERNAME")
    password = get_env("NEO4J_PASSWORD")
    google_api_key = get_env("GOOGLE_API_KEY")

    os.environ["GOOGLE_API_KEY"] = google_api_key

    # FIX: Added database=username for Neo4j Aura Free tier compatibility
    graph = Neo4jGraph(url=uri, username=username, password=password, database=username)
    graph.refresh_schema()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        convert_system_message_to_human=True,
    )

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=(
            "You are an expert Neo4j Cypher generator for the DataOnTips SAP Order-to-Cash graph.\n"
            "Use only the provided schema and relationship directions.\n"
            "Never invent labels, relationship types, or properties.\n"
            "Return only a valid Cypher query.\n\n"
            "Critical business patterns:\n"
            "- Ranking products by billed volume follows: "
            "(o:Order)-[:CONTAINS]->(p:Product), "
            "(o)-[:FULFILLED_BY]->(d:Delivery), "
            "(d)-[:BILLED_IN]->(b:BillingDocument)\n"
            "- Flow tracing for billing document follows reverse chain from "
            "BillingDocument <-[:BILLED_IN]- Delivery <-[:FULFILLED_BY]- Order.\n"
            "- Delivered but not billed means orders with a Delivery but without BillingDocument.\n\n"
            "Schema:\n{schema}\n\n"
            "Question:\n{question}\n"
        ),
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a grounded analyst for DataOnTips.\n"
            "Answer using ONLY the provided context from the Neo4j graph query results.\n"
            "If context is insufficient, say exactly: "
            "'I could not find enough data in the graph to answer that.'\n"
            "Do not add external facts.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n"
        ),
    )

    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=False,
        return_intermediate_steps=True,
        top_k=30,
        allow_dangerous_requests=True,
    )

    return graph, chain


def is_domain_query(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return False
    if any(token in q for token in DOMAIN_KEYWORDS):
        return True
    return False


def detect_billing_doc_id(question: str) -> str:
    pattern = r"(?:billing\s+document|invoice)\s*[:#]?\s*([A-Za-z0-9_\-]+)"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def is_ranking_query(question: str) -> bool:
    q = question.lower()
    return "highest" in q and "product" in q and "billing" in q


def is_flow_trace_query(question: str) -> bool:
    q = question.lower()
    return "trace" in q and "flow" in q and ("billing document" in q or "invoice" in q)


def is_anomaly_query(question: str) -> bool:
    q = question.lower()
    return ("delivered" in q and "not yet billed" in q) or (
        "delivered" in q and "not billed" in q
    )


def run_ranked_products_query(graph: Neo4jGraph) -> Tuple[str, List[Dict]]:
    cypher = """
    MATCH (o:Order)-[:CONTAINS]->(p:Product)
    MATCH (o)-[:FULFILLED_BY]->(:Delivery)-[:BILLED_IN]->(b:BillingDocument)
    RETURN p.product_id AS product_id,
           coalesce(p.product_name, '') AS product_name,
           count(DISTINCT b) AS billing_document_count
    ORDER BY billing_document_count DESC, product_id ASC
    LIMIT 10
    """
    rows = graph.query(cypher)
    if not rows:
        return "No ranking data found in the graph.", rows
    lines = [
        f"{idx + 1}. Product `{r.get('product_id', '')}` "
        f"({r.get('product_name', '')}) -> {r.get('billing_document_count', 0)} billing docs"
        for idx, r in enumerate(rows)
    ]
    return "Top products by number of billing documents:\n\n" + "\n".join(lines), rows


def run_flow_trace_query(graph: Neo4jGraph, billing_id: str) -> Tuple[str, List[Dict]]:
    cypher = """
    MATCH (b:BillingDocument {billing_id: $billing_id})
    OPTIONAL MATCH (d:Delivery)-[:BILLED_IN]->(b)
    OPTIONAL MATCH (o:Order)-[:FULFILLED_BY]->(d)
    RETURN b.billing_id AS billing_id,
           d.delivery_id AS delivery_id,
           o.order_id AS order_id
    LIMIT 25
    """
    rows = graph.query(cypher, params={"billing_id": billing_id})
    rows = [r for r in rows if any([r.get("billing_id"), r.get("delivery_id"), r.get("order_id")])]
    if not rows:
        return f"No flow found for billing document `{billing_id}`.", rows
    flow_lines = [
        f"Sales Order `{r.get('order_id', 'N/A')}` -> Delivery `{r.get('delivery_id', 'N/A')}` -> Billing `{r.get('billing_id', 'N/A')}`"
        for r in rows
    ]
    return f"Flow trace for billing document `{billing_id}`:\n\n" + "\n".join(flow_lines), rows


def run_anomaly_query(graph: Neo4jGraph) -> Tuple[str, List[Dict]]:
    cypher = """
    MATCH (o:Order)-[:FULFILLED_BY]->(d:Delivery)
    WHERE NOT (d)-[:BILLED_IN]->(:BillingDocument)
    RETURN DISTINCT o.order_id AS order_id,
           d.delivery_id AS delivery_id
    ORDER BY o.order_id
    LIMIT 50
    """
    rows = graph.query(cypher)
    if not rows:
        return "No delivered-but-not-billed anomalies were found.", rows
    lines = [f"- Order `{r.get('order_id', '')}` with Delivery `{r.get('delivery_id', '')}`" for r in rows]
    return "Delivered but not yet billed sales orders:\n\n" + "\n".join(lines), rows


def run_custom_business_tool(graph: Neo4jGraph, question: str) -> Tuple[bool, str]:
    if is_ranking_query(question):
        answer, _ = run_ranked_products_query(graph)
        return True, answer

    if is_flow_trace_query(question):
        billing_id = detect_billing_doc_id(question)
        if not billing_id:
            return True, "Please provide a billing document ID, e.g., 'Trace the flow for billing document 90012345'."
        answer, _ = run_flow_trace_query(graph, billing_id)
        return True, answer

    if is_anomaly_query(question):
        answer, _ = run_anomaly_query(graph)
        return True, answer

    return False, ""


def ask_grounded_chain(chain: GraphCypherQAChain, question: str) -> str:
    result = chain.invoke({"query": question})
    answer = result.get("result", "").strip()
    if not answer:
        return "I could not find enough data in the graph to answer that."
    return answer


def fetch_graph_preview(graph: Neo4jGraph, limit: int = MAX_GRAPH_RELATIONS):
    cypher = """
    MATCH (a)-[r]->(b)
    WITH a, r, b
    ORDER BY id(r) DESC
    LIMIT $limit
    RETURN
      labels(a)[0] AS source_label,
      coalesce(a.customer_id, a.order_id, a.product_id, a.delivery_id, a.billing_id, toString(id(a))) AS source_key,
      labels(b)[0] AS target_label,
      coalesce(b.customer_id, b.order_id, b.product_id, b.delivery_id, b.billing_id, toString(id(b))) AS target_key,
      type(r) AS rel_type
    """
    rows = graph.query(cypher, params={"limit": limit})

    node_registry = {}
    edges: List[Edge] = []

    for i, row in enumerate(rows):
        src_id = f"{row['source_label']}:{row['source_key']}"
        tgt_id = f"{row['target_label']}:{row['target_key']}"

        if src_id not in node_registry:
            node_registry[src_id] = Node(
                id=src_id,
                label=f"{row['source_label']}\n{row['source_key']}",
                color=NODE_COLORS.get(row["source_label"], "#9E9E9E"),
                size=22,
            )
        if tgt_id not in node_registry:
            node_registry[tgt_id] = Node(
                id=tgt_id,
                label=f"{row['target_label']}\n{row['target_key']}",
                color=NODE_COLORS.get(row["target_label"], "#9E9E9E"),
                size=22,
            )

        edges.append(
            Edge(
                source=src_id,
                target=tgt_id,
                label=row["rel_type"],
                color="#B0BEC5",
                smooth=True,
            )
        )

    return list(node_registry.values()), edges


def render_sidebar(connected: bool):
    st.sidebar.title("DataOnTips")
    st.sidebar.markdown("### Database Status")
    if connected:
        st.sidebar.success("Connected to Neo4j")
    else:
        st.sidebar.error("Not connected")
    st.sidebar.caption("Model: gemini-1.5-flash")

    if st.sidebar.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Grounded conversational analytics over SAP Order-to-Cash graph data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        graph, chain = init_services()
        connected = True
    except Exception as exc:
        connected = False
        graph, chain = None, None
        st.error(f"Initialization failed: {exc}")

    render_sidebar(connected)
    if not connected:
        st.stop()

    left_col, right_col = st.columns([1.05, 1.2], gap="large")

    with left_col:
        st.subheader("Graph Visualization")
        with st.spinner("Loading graph preview..."):
            nodes, edges = fetch_graph_preview(graph, limit=MAX_GRAPH_RELATIONS)
        config = Config(
            width="100%",
            height=680,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=True,
        )
        if nodes and edges:
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.info("No graph connections found yet. Run ingestion and refresh.")

    with right_col:
        st.subheader("Conversational Query Interface")
        st.markdown(
            "Ask business questions about customers, orders, deliveries, products, and billing documents."
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about the SAP O2C dataset...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Querying graph and generating grounded answer..."):
                    if not is_domain_query(user_input):
                        response = DEFAULT_REJECTION
                    else:
                        handled, custom_response = run_custom_business_tool(graph, user_input)
                        if handled:
                            response = custom_response
                        else:
                            try:
                                response = ask_grounded_chain(chain, user_input)
                            except Exception as exc:
                                response = f"Query failed: {exc}"
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()