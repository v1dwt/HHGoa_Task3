# Hacker House Goa 2026 — Task #3

## Face Identification & Blockchain Verification

This project implements a face-identification pipeline that:

1. Takes a face image as input.
2. Generates a face embedding using InsightFace.
3. Performs a genuine reverse-image search using Google Lens through SerpApi.
4. Dynamically discovers social-media posts from the search results.
5. Downloads candidate images from discovered posts.
6. Verifies the candidate faces against the input face using cosine similarity.
7. Stores the discovered post data and its SHA-256 fingerprint in a local blockchain.
8. Recalculates the hash to verify that the stored data has not been modified.

No social-media post URL is hardcoded into the search pipeline.

---

## Architecture

```text
Input Face
    |
    v
InsightFace
    |
    | Face embedding
    v
SerpApi Image Upload
    |
    v
Google Lens Reverse Image Search
    |
    v
Dynamic Social-Media Discovery
    |
    v
Candidate Image Extraction
    |
    v
InsightFace Face Verification
    |
    | Cosine similarity
    v
Best Matching Social Post
    |
    v
SHA-256 Hash
    |
    v
Local Blockchain
    |
    v
Hash Re-verification