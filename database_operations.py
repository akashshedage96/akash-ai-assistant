from urllib.parse import quote_plus

from sqlalchemy import create_engine
from config import SQL_DRIVER, SQL_SERVER, SQL_DATABASE


def get_connection_url(database):
    conn_str = (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
    )

    return "mssql+pyodbc:///?odbc_connect=" + quote_plus(conn_str)


def ensure_database():
    master_engine = create_engine(
        get_connection_url("master"),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )

    try:
        safe_name = SQL_DATABASE.replace("]", "]]")
        safe_literal = SQL_DATABASE.replace("'", "''")

        with master_engine.connect() as conn:
            conn.exec_driver_sql(
                f"""
                IF DB_ID(N'{safe_literal}') IS NULL
                BEGIN
                    CREATE DATABASE [{safe_name}]
                END
                """
            )
    finally:
        master_engine.dispose()


def create_database_engine():
    ensure_database()

    return create_engine(
        get_connection_url(SQL_DATABASE),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        fast_executemany=True,
    )