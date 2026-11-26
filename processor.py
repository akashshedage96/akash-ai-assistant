from typing import Optional


class BaseProcessor:
    def process(self, blob_name: str, content: bytes, etag: Optional[str]) -> list[str]:
        raise NotImplementedError

    def delete(self, blob_name: str, stored_ids: list[str]) -> None:
        raise NotImplementedError