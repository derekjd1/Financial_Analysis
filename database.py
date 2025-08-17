# new.py — SQLite helpers for favorites list

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from typing import Optional

import pandas as pd
import streamlit as st

DB_PATH = os.environ.get("FINANCE_APP_DB", "favorites.db")

@st.cache_resource(show_spinner=False)
def get_conn(path: str = DB_PATH) -> sqlite3.Connection:
    """Create/return a cached SQLite connection and ensure schema exists."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn

def add_favorite(conn: sqlite3.Connection, symbol: str, label: Optional[str]) -> None:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return
    lbl = (label or "").strip() or None
    with closing(conn.cursor()) as cur:
        cur.execute(
            "INSERT OR IGNORE INTO favorites(symbol, label) VALUES(?, ?)",
            (symbol, lbl),
        )
        conn.commit()

def remove_favorite(conn: sqlite3.Connection, symbol: str) -> None:
    with closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM favorites WHERE symbol = ?", ((symbol or "").strip().upper(),))
        conn.commit()

def list_favorites(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id, symbol, COALESCE(label,'') AS label, created_at FROM favorites ORDER BY symbol ASC",
        conn,
    )
