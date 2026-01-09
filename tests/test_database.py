import sqlite3
import pandas as pd

from database import get_conn, add_favorite, remove_favorite, list_favorites

# unit test for the database file

def test_add_and_list_favorites(tmp_path):
    db_path = tmp_path / "test_favs.db"

    conn = get_conn(str(db_path))

    add_favorite(conn, "AAPL", "Apple")
    add_favorite(conn, "VFV.TO", "Vanguard S&P 500")

    df = list_favorites(conn)

    assert isinstance(df, pd.DataFrame)
    df_str = df.astype(str)
    assert df_str.apply(lambda col: col.str.contains("AAPL", case=False, na=False)).any().any()
    assert df_str.apply(lambda col: col.str.contains("VFV.TO", case=False, na=False)).any().any()


def test_remove_favorite(tmp_path):
    db_path = tmp_path / "test_favs.db"
    conn = get_conn(str(db_path))

    add_favorite(conn, "AAPL", "Apple")
    remove_favorite(conn, "AAPL")

    df = list_favorites(conn)

    df_str = df.astype(str)
    assert not df_str.apply(lambda col: col.str.contains("AAPL", case=False, na=False)).any().any()
