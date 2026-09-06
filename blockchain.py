import hashlib
import json
from datetime import datetime, timezone


class LocalBlockchain:
    """Simple local blockchain simulation for storing and verifying data hashes."""

    def __init__(self):
        """Initialize an empty blockchain."""
        self.chain = []

    def calculate_hash(self, data):
        """
        Calculate a deterministic SHA-256 hash for the supplied data.

        Sorting dictionary keys and using fixed JSON separators ensures
        equivalent data produces the same hash.
        """
        canonical_data = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":")
        )

        return hashlib.sha256(
            canonical_data.encode("utf-8")
        ).hexdigest()

    def upload(self, data):
        """
        Store data in a new block and return the created block.
        """
        data_hash = self.calculate_hash(data)

        block = {
            "index": len(self.chain),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_hash": data_hash,
            "data": data
        }

        self.chain.append(block)

        return block

    def verify(self, block):
        """
        Verify that the stored data has not changed since upload.
        """
        stored_hash = block["data_hash"]
        recalculated_hash = self.calculate_hash(block["data"])

        return stored_hash == recalculated_hash


if __name__ == "__main__":

    blockchain = LocalBlockchain()

    test_data = {
        "source": "Facebook",
        "title": "Test social media post",
        "url": "https://example.com/post",
        "face_similarity": 0.78
    }

    block = blockchain.upload(test_data)

    print("\n========================================")
    print("BLOCKCHAIN RECORD")
    print("========================================")

    print(json.dumps(block, indent=2))

    print("\n========================================")
    print("BLOCKCHAIN VERIFICATION")
    print("========================================")

    print("Verification:", blockchain.verify(block))