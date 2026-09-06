# Hacker House Goa 2026 — Task #3
## Face Identification & Blockchain Verification

An end-to-end pipeline that takes a face image, performs genuine reverse-image search to discover matching web/social-media content, verifies the discovered content using face recognition, and stores a tamper-evident hash on a local blockchain.

---

## What This Project Does

The pipeline follows this flow:

Face Image
↓
Face Detection & Encoding
↓
Google Lens Reverse Image Search
↓
Web / Social Media Candidate Discovery
↓
Candidate Image Extraction
↓
Face Similarity Verification
↓
Matching Social Media Post
↓
SHA-256 Hash
↓
Local Blockchain
↓
Blockchain Re-verification

The project demonstrates the complete flow required for Hacker House Goa 2026 Task #3.

---

## Task Requirements Covered

### 1. Face Identification

The project uses **InsightFace** with the `buffalo_l` model to:

- Detect faces in images
- Generate face embeddings
- Compare embeddings using cosine similarity
- Determine whether a candidate image contains the same person

### 2. Web / Social Media Search

The project uses **SerpApi Google Lens** for genuine reverse-image search.

The input image is uploaded to SerpApi and Google Lens results are retrieved dynamically.

The implementation does not hardcode or pre-select the final social-media post.

The pipeline:

1. Uploads the input image to SerpApi
2. Performs a Google Lens search
3. Collects visual and organic results
4. Filters results for supported social-media domains
5. Extracts candidate images from discovered posts
6. Downloads and verifies candidate images
7. Runs face recognition against the input image
8. Selects a matching post based on face similarity

Supported social-media sources include:

- Instagram
- Facebook
- X / Twitter
- TikTok
- LinkedIn
- YouTube

### 3. Blockchain Verification

The project uses a **local simulated blockchain**.

After a matching post is discovered, the pipeline creates a deterministic SHA-256 hash from the discovered post metadata:

```json
{
  "source": "Facebook",
  "title": "Example post title",
  "url": "https://example.com/post",
  "face_similarity": 0.63
}