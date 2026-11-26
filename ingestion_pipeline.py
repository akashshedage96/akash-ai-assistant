import logging

logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(self, blob_service, state_store, processors):
        self.blobs = blob_service
        self.state = state_store
        self.processors = processors     

    def run(self):
        known = self.state.all_files()

        # If listing fails this raises
        
        files = list(self.blobs.collect_files())
        seen = {f["name"] for f in files}

        logger.info(
            "Ingestion started: %d files in blob storage, %d already processed before",
            len(files), len(known),
        )

        counts = {"new": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0}

        #  New or changed files
        for f in files:
            name = f["name"]
            old = known.get(name)

            if old and old["etag"] == f["etag"]:
                logger.info("UNCHANGED (skipped): %s", name)
                counts["unchanged"] += 1
                continue

            if old:
                logger.info("UPDATED file (etag changed): %s", name)
            else:
                logger.info("NEW file: %s", name)

            processor = self.processors[f["extension"]]

            try:
                content = self.blobs.download_file(name)

                if old:                                     # changed: remove old data first
                    processor.delete(name, old["vector_ids"])
                    logger.info("  removed %d old keys for %s", len(old["vector_ids"]), name)

                ids = processor.process(name, content, f["etag"])
                self.state.save(name, f["etag"], f["extension"], ids)

                logger.info("  saved %d items for %s", len(ids), name)
                counts["updated" if old else "new"] += 1
            except Exception:
                logger.exception("FAILED to process %s", name)
                counts["failed"] += 1

        # Files deleted from blob storage
        for name, old in known.items():
            if name in seen:
                continue
            try:
                logger.info("DELETED from blob storage: %s", name)
                processor = self.processors.get(old["file_type"])
                if processor:
                    processor.delete(name, old["vector_ids"])
                self.state.delete(name)
                logger.info("  removed %d items for %s", len(old["vector_ids"]), name)
                counts["removed"] += 1
            except Exception:
                logger.exception("FAILED to remove %s", name)
                counts["failed"] += 1

        logger.info(
            "Ingestion finished: %d new, %d updated, %d unchanged, %d removed, %d failed",
            counts["new"], counts["updated"], counts["unchanged"],
            counts["removed"], counts["failed"],
        )