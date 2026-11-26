
import json
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from config import SQL_DRIVER, SQL_SERVER, SQL_DATABASE


class IngestionStateStore:

    def __init__(self,engine):
        # Create the database if it does not exist
        
        self.engine = engine
        # Create the tracking table if it does not exist
        try:
            self._ensure_table()
        except Exception:
            self.engine.dispose()
            raise

    
    
    

    def _ensure_table(self):

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    IF OBJECT_ID(N'dbo.file_state', N'U') IS NULL
                    BEGIN
                        CREATE TABLE dbo.file_state (
                            blob_name NVARCHAR(450) NOT NULL
                                PRIMARY KEY,
                            etag NVARCHAR(200) NOT NULL,
                            file_type NVARCHAR(50) NOT NULL,
                            vector_ids NVARCHAR(MAX) NOT NULL
                                DEFAULT '[]',
                            processed_at DATETIME2 NOT NULL
                                DEFAULT SYSUTCDATETIME()
                        )
                    END
                """)
            )

    # ---------- get one file ----------

    def get(self, blob_name):
        """Return the processing state of one file."""

        with self.engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT etag, file_type, vector_ids
                    FROM dbo.file_state
                    WHERE blob_name = :blob_name
                """),
                {"blob_name": blob_name},
            ).mappings().first()

        if row is None:
            return None

        return {
            "etag": row["etag"],
            "file_type": row["file_type"],
            "vector_ids": json.loads(row["vector_ids"]),
        }

    # ---------- insert or update ----------

    def save(self, blob_name, etag, file_type, vector_ids):
        """Insert a new file record or update an existing one."""

        ids = json.dumps(vector_ids)

        with self.engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE dbo.file_state
                    SET etag = :etag,
                        file_type = :file_type,
                        vector_ids = :vector_ids,
                        processed_at = SYSUTCDATETIME()
                    WHERE blob_name = :blob_name
                """),
                {
                    "blob_name": blob_name,
                    "etag": etag,
                    "file_type": file_type,
                    "vector_ids": ids,
                },
            )

            if result.rowcount == 0:
                conn.execute(
                    text("""
                        INSERT INTO dbo.file_state
                            (blob_name, etag, file_type, vector_ids)
                        VALUES
                            (:blob_name, :etag, :file_type, :vector_ids)
                    """),
                    {
                        "blob_name": blob_name,
                        "etag": etag,
                        "file_type": file_type,
                        "vector_ids": ids,
                    },
                )

    # ---------- get all files ----------

    def all_files(self):
        """Return all tracked files as a dictionary."""

        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT blob_name, etag, file_type, vector_ids
                    FROM dbo.file_state
                """)
            ).mappings().all()

        return {
            row["blob_name"]: {
                "etag": row["etag"],
                "file_type": row["file_type"],
                "vector_ids": json.loads(row["vector_ids"]),
            }
            for row in rows
        }

    # ---------- delete one file ----------

    def delete(self, blob_name):
        """Delete a file's tracking record."""

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    DELETE FROM dbo.file_state
                    WHERE blob_name = :blob_name
                """),
                {"blob_name": blob_name},
            )

    # ---------- cleanup ----------

    def close(self):

        self.engine.dispose()


