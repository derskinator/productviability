# graph_store.py

from typing import Optional
import pandas as pd
from neo4j import GraphDatabase


class GraphStore:
    """
    Simple wrapper around Neo4j driver for this project.
    Handles:
      - connecting to the database
      - wiping the graph (optional)
      - inserting Product and Keyword nodes
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def reset(self):
        """Delete all nodes and relationships. Use carefully."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_products(self, df_products: pd.DataFrame):
        """
        Expects columns:
          product (or name), keywords, branded_share, avg_volume,
          avg_comp, avg_cpc, avg_intent, avg_score
        """
        query = """
        MERGE (p:Product {name: $name})
        SET p.keywords      = $keywords,
            p.branded_share = $branded_share,
            p.avg_volume    = $avg_volume,
            p.avg_comp      = $avg_comp,
            p.avg_cpc       = $avg_cpc,
            p.avg_intent    = $avg_intent,
            p.avg_score     = $avg_score
        """
        with self.driver.session() as session:
            for _, row in df_products.iterrows():
                params = {
                    "name":          row["product"],
                    "keywords":      float(row["keywords"]),
                    "branded_share": float(row["branded_share"]),
                    "avg_volume":    float(row["avg_volume"]),
                    "avg_comp":      float(row["avg_comp"]),
                    "avg_cpc":       float(row["avg_cpc"]),
                    "avg_intent":    float(row["avg_intent"]),
                    "avg_score":     float(row["avg_score"]),
                }
                session.run(query, **params)

    def upsert_keywords(self, df_keywords: pd.DataFrame):
        """
        Expects columns:
          product, keyword, search_volume_raw, competition_raw, cpc_raw,
          is_branded, intent_score, viability_score, difficulty_bucket
        """
        query = """
        MERGE (p:Product {name: $product})
        MERGE (k:Keyword {text: $keyword})
        SET k.search_volume_raw  = $search_volume_raw,
            k.competition_raw    = $competition_raw,
            k.cpc_raw            = $cpc_raw,
            k.is_branded         = $is_branded,
            k.intent_score       = $intent_score,
            k.viability_score    = $viability_score,
            k.difficulty_bucket  = $difficulty_bucket
        MERGE (k)-[:BELONGS_TO]->(p)
        """
        with self.driver.session() as session:
            for _, row in df_keywords.iterrows():
                params = {
                    "product":           row["product"],
                    "keyword":           row["keyword"],
                    "search_volume_raw": float(row["search_volume_raw"])
                                          if pd.notna(row["search_volume_raw"]) else None,
                    "competition_raw":   float(row["competition_raw"])
                                          if pd.notna(row["competition_raw"]) else None,
                    "cpc_raw":           float(row["cpc_raw"])
                                          if pd.notna(row["cpc_raw"]) else None,
                    "is_branded":        bool(row["is_branded"]),
                    "intent_score":      float(row["intent_score"]),
                    "viability_score":   float(row["viability_score"]),
                    "difficulty_bucket": str(row["difficulty_bucket"]),
                }
                session.run(query, **params)
