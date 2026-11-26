import hashlib
import io
import json
import logging
import re

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mssql import BIGINT, BIT, DATETIME2, DECIMAL, NVARCHAR


logger = logging.getLogger(__name__)
class StructuredFileIngestor:
    def __init__(self,engine):
        self.engine = engine

    def get_table_name(self,blob_name: str) -> str:
        digest = hashlib.sha256(blob_name.encode("utf-8")).hexdigest()[:20]
        return f"file_{digest}"
    def create_metadata_table(self):
        query = """
        IF OBJECT_ID('dbo.structured_file_metadata', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.structured_file_metadata (
                serial_id INT IDENTITY(1,1) PRIMARY KEY,
                blob_name NVARCHAR(1024) NOT NULL,
                table_name NVARCHAR(128) NOT NULL,
                columns_json NVARCHAR(MAX) NOT NULL,
                updated_at DATETIME2
                    DEFAULT SYSUTCDATETIME()
            );
        END;
        """

        with self.engine.begin() as conn:
            conn.execute(text(query))

    def insert_into_metadata_table(self,blob_name: str, table_name: str, columns_json: list) -> None:
       
        query = """
        INSERT INTO dbo.structured_file_metadata
            (blob_name, table_name, columns_json)
        VALUES
            (:blob_name, :table_name, :columns_json);
        """

        with self.engine.begin() as conn:
            conn.execute(
                text(query),
                {
                    "blob_name": blob_name,
                    "table_name": table_name,
                    "columns_json": json.dumps(columns_json),
                },
            )
    def clean_dataframe(self,df: pd.DataFrame) -> pd.DataFrame:
        
        columns = []

        for column in df.columns:
            base = re.sub(
                r"[^a-zA-Z0-9_]+", "_", str(column).strip().lower()
            ).strip("_")
            base = (base)[:100]   

            name = base
        
        
            columns.append(name)

        df = df.copy()
        df.columns = columns
        logger.info("Cleaned dataframe columns: %s", df.columns.tolist())
        return df


    def ingest_structured_file(self,blob_name: str, content: bytes) -> str:
        name = blob_name.lower()

        if name.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="cp1252")

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))   # first sheet only

        else:
            raise ValueError(f"Unsupported structured file: {blob_name}")

        if len(df.columns) == 0:
            raise ValueError(f"No columns found in {blob_name}")

        df = self.clean_dataframe(df)

        # Text columns: store as NVARCHAR(MAX) so Hindi/Chinese/etc are kept
        # and convert mixed values (numbers + text) to plain strings
        dtype = {}
        for col in df.columns:
            if (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            ):
                # Text columns
                df[col] = df[col].map(
                    lambda v: None if pd.isna(v) else str(v)
                )
                dtype[col] = NVARCHAR()

            elif pd.api.types.is_bool_dtype(df[col]):
                # True / False columns
                dtype[col] = BIT()

            elif pd.api.types.is_integer_dtype(df[col]):
                # Whole number columns
                dtype[col] = BIGINT()

            elif pd.api.types.is_float_dtype(df[col]):
                # Decimal number columns
                dtype[col] = DECIMAL(18, 4)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Date and time columns
                dtype[col] = DATETIME2()
        table_name = self.get_table_name(blob_name)

        # One transaction: if loading fails, the old table is kept
        with self.engine.begin() as conn:
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists="replace",
                index=False,
                chunksize=1000,
                dtype=dtype,
            )
        self.create_metadata_table()
        self.insert_into_metadata_table(blob_name, table_name, df.columns.tolist())
        logger.info("SQL updated: %s -> %s (%d rows)", blob_name, table_name, len(df))
        return table_name



    def delete_structured_file(self,blob_name: str) -> None:
        table_name = self.get_table_name(blob_name)
        quoted = self.engine.dialect.identifier_preparer.quote(table_name)   

        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {quoted}"))
            conn.execute(
                        text("""
                            DELETE FROM dbo.structured_file_metadata
                            WHERE blob_name = :blob_name
                        """),
                        {"blob_name": blob_name},
                    )

        logger.info("Dropped table and its metadata information %s for %s", table_name, blob_name)
     