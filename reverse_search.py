from blockchain import LocalBlockchain
import os
import requests
import cv2
import numpy as np
import json
from html import unescape

from dotenv import load_dotenv
from insightface.app import FaceAnalysis


# =========================================================
# 1. Load API key
# =========================================================

load_dotenv()

API_KEY = os.getenv("SERPAPI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "SERPAPI_API_KEY was not found in the .env file."
    )


# =========================================================
# 2. Initialize InsightFace
# =========================================================

print("Initializing InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("InsightFace initialized successfully.\n")


# =========================================================
# 3. Input image
# =========================================================

INPUT_IMAGE = (
    r"dataset\lfw-deepfunneled\lfw-deepfunneled"
    r"\Aaron_Peirsol\Aaron_Peirsol_0001.jpg"
)

if not os.path.exists(INPUT_IMAGE):
    raise FileNotFoundError(
        f"Input image not found: {INPUT_IMAGE}"
    )


# =========================================================
# 4. Generate face embedding
# =========================================================

def get_embedding_from_file(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    faces = face_app.get(image)

    if len(faces) == 0:
        raise RuntimeError(
            f"No face detected in: {image_path}"
        )

    return faces[0].embedding


print("Generating input face embedding...")

input_embedding = get_embedding_from_file(INPUT_IMAGE)

print(
    f"Input embedding length: {len(input_embedding)}"
)


# =========================================================
# 5. Upload input image to SerpApi
# =========================================================

print("\nUploading image to SerpApi...")

upload_url = "https://serpapi.com/image"

with open(INPUT_IMAGE, "rb") as image_file:
    upload_response = requests.post(
        upload_url,
        files={
            "image": (
                os.path.basename(INPUT_IMAGE),
                image_file,
                "image/jpeg"
            )
        },
        data={
            "api_key": API_KEY
        },
        timeout=60
    )

print(
    f"Upload HTTP status: "
    f"{upload_response.status_code}"
)

upload_response.raise_for_status()

upload_data = upload_response.json()

image_id = upload_data.get("image_id")

if not image_id:
    raise RuntimeError(
        f"No image_id returned: {upload_data}"
    )

print(f"Image ID received: {image_id}")


# =========================================================
# 6. Google Lens search
# =========================================================

print("\nSearching Google Lens...")

# Derive the person name dynamically from the input-image folder.
# Example:
# ...\Aaron_Peirsol\Aaron_Peirsol_0001.jpg
# becomes:
# Aaron Peirsol
person_name = os.path.basename(
    os.path.dirname(INPUT_IMAGE)
).replace("_", " ")

print(f"Lens name refinement: {person_name}")

lens_results = []


# ---------------------------------------------------------
# Search 1: Standard Google Lens
# ---------------------------------------------------------

print("\nGoogle Lens search: standard")

standard_response = requests.get(
    "https://serpapi.com/search.json",
    params={
        "engine": "google_lens",
        "image_id": image_id,
        "type": "all",
        "api_key": API_KEY
    },
    timeout=60
)

print(
    f"Google Lens HTTP status: "
    f"{standard_response.status_code}"
)

standard_response.raise_for_status()

standard_data = standard_response.json()

if "error" in standard_data:
    print(
        "Standard Lens search returned an error:",
        standard_data["error"]
    )
else:
    lens_results.extend(
        standard_data.get("visual_matches", [])
    )
    lens_results.extend(
        standard_data.get("organic_results", [])
    )


# ---------------------------------------------------------
# Search 2: Auto-crop Google Lens
# ---------------------------------------------------------

print("\nGoogle Lens search: auto_crop")

auto_crop_response = requests.get(
    "https://serpapi.com/search.json",
    params={
        "engine": "google_lens",
        "image_id": image_id,
        "type": "all",
        "auto_crop": "true",
        "api_key": API_KEY
    },
    timeout=60
)

print(
    f"Google Lens HTTP status: "
    f"{auto_crop_response.status_code}"
)

auto_crop_response.raise_for_status()

auto_crop_data = auto_crop_response.json()

if "error" in auto_crop_data:
    print(
        "Auto-crop Lens search returned an error:",
        auto_crop_data["error"]
    )
else:
    lens_results.extend(
        auto_crop_data.get("visual_matches", [])
    )
    lens_results.extend(
        auto_crop_data.get("organic_results", [])
    )


# ---------------------------------------------------------
# Search 3: Image + dynamically derived person name
# ---------------------------------------------------------

print(
    "\nGoogle Lens search: "
    "person-name assisted"
)

name_response = requests.get(
    "https://serpapi.com/search.json",
    params={
        "engine": "google_lens",
        "image_id": image_id,
        "type": "all",
        "q": person_name,
        "api_key": API_KEY
    },
    timeout=60
)

print(
    f"Google Lens HTTP status: "
    f"{name_response.status_code}"
)

name_response.raise_for_status()

name_data = name_response.json()

if "error" in name_data:
    print(
        "Name-assisted Lens search returned an error:",
        name_data["error"]
    )
else:
    lens_results.extend(
        name_data.get("visual_matches", [])
    )
    lens_results.extend(
        name_data.get("organic_results", [])
    )


# ---------------------------------------------------------
# Deduplicate all Lens results
# ---------------------------------------------------------

all_lens_results = []

seen_links = set()

for match in lens_results:
    link = match.get("link", "")

    if link:
        if link in seen_links:
            continue

        seen_links.add(link)

    all_lens_results.append(match)


# ---------------------------------------------------------
# Save combined Lens response
# ---------------------------------------------------------

combined_lens_data = {
    "standard_results": standard_data,
    "auto_crop_results": auto_crop_data,
    "name_assisted_results": name_data,
    "total_unique_results": len(all_lens_results)
}

with open(
    "lens_response.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        combined_lens_data,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    "\nPrimary Lens response saved to "
    "lens_response.json"
)

print(
    f"Total unique Lens results available: "
    f"{len(all_lens_results)}"
)

# =========================================================
# 7. Find specific social-media posts
# =========================================================

SOCIAL_DOMAINS = [
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "linkedin.com",
    "youtube.com"
]

POST_PATTERNS = [
    "/p/",
    "/reel/",
    "/posts/",
    "/post/",
    "/video/",
    "/videos/",
    "/watch",
    "/status/",
    "/photo/",
    "/photos/",
    "/shorts/"
]

social_results = []

for match in all_lens_results:
    link = match.get("link", "")

    if not link:
        continue

    link_lower = link.lower()

    is_social = any(
        domain in link_lower
        for domain in SOCIAL_DOMAINS
    )

    # Check whether this is a specific social-media post
    is_specific_post = any(
        pattern in link_lower
        for pattern in POST_PATTERNS
    )

    if not is_social:
     continue

    if not is_specific_post:
     continue

    social_results.append(match)


print("\n========================================")
print("SOCIAL MEDIA POST RESULTS")
print("========================================")

print(
    f"Specific social-media posts found: "
    f"{len(social_results)}"
)

for index, match in enumerate(
    social_results,
    start=1
):
    print("\n----------------------------------------")
    print(f"Post #{index}")

    print(
        "Title:",
        match.get("title", "N/A")
    )

    print(
        "Source:",
        match.get("source", "N/A")
    )

    print(
        "Link:",
        match.get("link", "N/A")
    )

    print(
        "Image:",
        match.get("image", "N/A")
    )


# =========================================================
# 8. Find and verify social-media candidates
# =========================================================

print("\n========================================")
print("VERIFYING SOCIAL MEDIA CANDIDATES")
print("========================================")


def get_candidate_image_urls(match):
    image_urls = []
    """
    Return possible image URLs for a dynamically discovered
    social-media post.

    URLs are tried in this order:
    1. Lens-provided image URL
    2. High-resolution image URLs discovered from post HTML
    3. Open Graph / Twitter image metadata
    4. Lens thumbnail
    """

    import re
    from html import unescape
    from urllib.parse import unquote
        # Google Lens can return a usable thumbnail even when the
    # social-media page itself does not expose its image URL.
    lens_image_fields = [
        "image",
        "thumbnail",
        "original",
        "original_image"
    ]

    for field in lens_image_fields:
        value = match.get(field)

        if isinstance(value, str) and value.startswith("http"):
            image_urls.append(value)

        elif isinstance(value, dict):
            for nested_field in ["url", "link", "src"]:
                nested_value = value.get(nested_field)

                if (
                    isinstance(nested_value, str)
                    and nested_value.startswith("http")
                ):
                    image_urls.append(nested_value)


    def add_image_url(url):
        if not url:
            return

        url = unescape(url)
        url = unquote(url)
        url = url.replace("\\/", "/")

        # Remove JSON escaping
        url = url.replace("\\u0026", "&")
        url = url.replace("\\u003D", "=")
        url = url.replace("\\u002F", "/")

        if url.startswith("//"):
            url = "https:" + url

        if url.startswith("http") and url not in image_urls:
            image_urls.append(url)

    # -----------------------------------------------------
    # 1. Lens-provided image
    # -----------------------------------------------------

    direct_image = match.get("image")

    if direct_image:
        add_image_url(direct_image)

    # -----------------------------------------------------
    # 2. Retrieve the actual social-media post HTML
    # -----------------------------------------------------

    post_url = match.get("link", "")

    if post_url:
        try:
            print(
                "Trying to retrieve image metadata "
                "from discovered post..."
            )

            page_response = requests.get(
                post_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/140.0.0.0 Safari/537.36"
                    ),
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,image/avif,"
                        "image/webp,*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.9"
                },
                timeout=30
            )

            print(
                f"Post page HTTP status: "
                f"{page_response.status_code}"
            )

            if page_response.status_code == 200:

                html = page_response.text

                # -------------------------------------------------
                # 2A. Open Graph / Twitter metadata
                # -------------------------------------------------

                metadata_patterns = [
                    r'<meta[^>]+property=["\']og:image(?::url)?["\'][^>]+content=["\']([^"\']+)["\']',
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::url)?["\']',
                    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']'
                ]

                for pattern in metadata_patterns:
                    matches_found = re.findall(
                        pattern,
                        html,
                        re.IGNORECASE
                    )

                    for image_url in matches_found:
                        add_image_url(image_url)

                # -------------------------------------------------
                # 2B. Extract Instagram CDN image URLs
                # -------------------------------------------------

                instagram_patterns = [
                    r'https?://scontent[^"\'\\\s<>]+',
                    r'https?:\\\\/\\\\/scontent[^"\'\\\s<>]+',
                    r'https?://[^"\'\\\s<>]*fbcdn\.net[^"\'\\\s<>]+',
                    r'https?:\\\\/\\\\/[^"\'\\\s<>]*fbcdn\.net[^"\'\\\s<>]+'
                ]

                for pattern in instagram_patterns:

                    matches_found = re.findall(
                        pattern,
                        html,
                        re.IGNORECASE
                    )

                    for image_url in matches_found:
                        add_image_url(image_url)

                # -------------------------------------------------
                # 2C. Extract image URLs from JSON fields
                # -------------------------------------------------

                json_patterns = [
                    r'"display_url":"([^"]+)"',
                    r'"thumbnail_src":"([^"]+)"',
                    r'"image_url":"([^"]+)"',
                    r'"image_versions2".*?"url":"([^"]+)"'
                ]

                for pattern in json_patterns:

                    matches_found = re.findall(
                        pattern,
                        html,
                        re.IGNORECASE
                    )

                    for image_url in matches_found:
                        add_image_url(image_url)

                # -------------------------------------------------
                # 2D. Facebook CDN URLs
                # -------------------------------------------------

                facebook_urls = re.findall(
                    r'https?://(?:scontent[^"\'\\\s]+|'
                    r'lookaside\.fbsbx\.com[^"\'\\\s]+)',
                    html,
                    re.IGNORECASE
                )

                for image_url in facebook_urls:
                    add_image_url(image_url)

                # -------------------------------------------------
                # Remove obviously tiny Instagram crawler URLs
                # -------------------------------------------------

                filtered_urls = []

                for image_url in image_urls:

                    # Google/Lens Instagram crawler images often
                    # contain "google_widget/crawler".
                    if (
                        "lookaside.instagram.com/seo/google_widget/crawler"
                        in image_url
                    ):
                        continue

                    if image_url not in filtered_urls:
                        filtered_urls.append(image_url)

                image_urls = filtered_urls

                if image_urls:
                    print(
                        f"Found {len(image_urls)} "
                        f"possible high-resolution image URL(s)."
                    )

        except Exception as e:

            print(
                f"Could not retrieve post image metadata: "
                f"{e}"
            )

    # -----------------------------------------------------
    # 3. Lens thumbnail as final fallback
    # -----------------------------------------------------

    thumbnail = match.get("thumbnail")

    if thumbnail:
        add_image_url(thumbnail)
    if image_urls:
        print(
            f"Image URLs discovered for post: "
            f"{len(image_urls)}"
        )

    return image_urls


# Load all reference images for the same person.
reference_embeddings = []

REFERENCE_DIR = os.path.dirname(INPUT_IMAGE)

for reference_file in sorted(
    os.listdir(REFERENCE_DIR)
):
    if not reference_file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    reference_path = os.path.join(
        REFERENCE_DIR,
        reference_file
    )

    reference_embedding = (
        get_embedding_from_file(
            reference_path
        )
    )

    if reference_embedding is not None:
        reference_embeddings.append(
            reference_embedding
        )

print(
    f"Reference face embeddings loaded: "
    f"{len(reference_embeddings)}"
)

if not reference_embeddings:
    print(
        "No reference face embeddings found."
    )
    raise SystemExit(0)


candidate = None
candidate_embedding = None
candidate_similarity = -1.0
candidate_path = "candidate_social.jpg"

# Minimum similarity required before a candidate
# can be considered a verified face match.
THRESHOLD = 0.60


# ---------------------------------------------------------
# Verify each dynamically discovered social-media post.
# ---------------------------------------------------------
for index, match in enumerate(
    social_results,
    start=1
):

    title = match.get(
        "title",
        "N/A"
    )

    source = match.get(
        "source",
        "N/A"
    )

    link = match.get(
        "link",
        ""
    )

    print("\n----------------------------------------")
    print(
        f"Checking candidate #{index}"
    )

    print(
        f"Title: {title}"
    )

    print(
        f"Source: {source}"
    )

    print(
        f"URL: {link}"
    )

    # -----------------------------------------------------
    # Get possible image URLs dynamically from the
    # discovered social-media post.
    # -----------------------------------------------------
    candidate_image_urls = (
        get_candidate_image_urls(
            match
        )
    )

    if not candidate_image_urls:
        print(
            "Skipping: no image URLs found."
        )
        continue

    candidate_processed = False

    # Track the strongest face found across ALL usable images
    # belonging to this social-media post.
    post_best_similarity = -1.0
    post_best_embedding = None
    post_best_image = None
    post_best_image_response = None

    # -----------------------------------------------------
    # Try every possible image URL.
    # -----------------------------------------------------
    for image_attempt, candidate_image_url in enumerate(
        candidate_image_urls,
        start=1
    ):

        print(
            f"Trying image URL #{image_attempt}..."
        )

        try:

            image_response = requests.get(
                candidate_image_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/140.0.0.0 Safari/537.36"
                    )
                },
                timeout=30
            )

            print(
                f"HTTP status: "
                f"{image_response.status_code}"
            )

            content_type = (
                image_response.headers.get(
                    "Content-Type",
                    ""
                )
            )

            print(
                f"Content type: "
                f"{content_type}"
            )

            print(
                f"Response size: "
                f"{len(image_response.content)} bytes"
            )

            if image_response.status_code != 200:
                print(
                    "Skipping this image URL."
                )
                continue

            if not content_type.lower().startswith(
                "image/"
            ):
                print(
                    "Skipping: response is not an image."
                )
                continue

            temp_path = (
                f"candidate_{index}_{image_attempt}.jpg"
            )

            with open(
                temp_path,
                "wb"
            ) as output_file:

                output_file.write(
                    image_response.content
                )

            print(
                f"Image saved temporarily as: "
                f"{temp_path}"
            )

            candidate_image = cv2.imread(
                temp_path
            )

            if candidate_image is None:
                print(
                    "Skipping: OpenCV could not "
                    "read image."
                )
                continue

            height, width = (
                candidate_image.shape[:2]
            )

            print(
                f"Image dimensions: "
                f"{width}x{height}"
            )

            if width < 160 or height < 160:
                print(
                    "Skipping: image is too small "
                    "for reliable face verification."
                )
                continue

            candidate_processed = True

            # -------------------------------------------------
            # Detect every face in THIS image.
            # -------------------------------------------------

            candidate_faces = face_app.get(
                candidate_image
            )

            print(
                f"Faces detected: "
                f"{len(candidate_faces)}"
            )

            if len(candidate_faces) == 0:
                print(
                    "No face detected in this image."
                )
                continue

            # -------------------------------------------------
            # Compare every detected face against every
            # Aaron Peirsol reference image.
            # -------------------------------------------------

            image_best_similarity = -1.0
            image_best_embedding = None

            for face_number, face in enumerate(
                candidate_faces,
                start=1
            ):

                embedding = face.embedding

                reference_similarities = []

                for reference_embedding in (
                    reference_embeddings
                ):

                    reference_similarity = (
                        np.dot(
                            reference_embedding,
                            embedding
                        )
                        /
                        (
                            np.linalg.norm(
                                reference_embedding
                            )
                            *
                            np.linalg.norm(
                                embedding
                            )
                        )
                    )

                    reference_similarities.append(
                        reference_similarity
                    )

                similarity = max(
                    reference_similarities
                )

                print(
                    f"Face #{face_number} "
                    f"best reference similarity: "
                    f"{similarity:.4f}"
                )

                if similarity > image_best_similarity:

                    image_best_similarity = similarity
                    image_best_embedding = embedding

            print(
                f"Best similarity for image #{image_attempt}: "
                f"{image_best_similarity:.4f}"
            )

            # -------------------------------------------------
            # Keep the strongest image from this post.
            # -------------------------------------------------

            if image_best_similarity > post_best_similarity:

                post_best_similarity = (
                    image_best_similarity
                )

                post_best_embedding = (
                    image_best_embedding
                )

                post_best_image = (
                    candidate_image.copy()
                )

                post_best_image_response = (
                    image_response
                )

                print(
                    "New strongest image found "
                    "for this social-media post."
                )

        except Exception as e:

            print(
                f"Error downloading/processing image: {e}"
            )

            continue

    # -----------------------------------------------------
    # No usable image.
    # -----------------------------------------------------

    if not candidate_processed:

        print(
            "No usable high-resolution image found "
            "for this social-media result."
        )

        continue

    # -----------------------------------------------------
    # Show the strongest result for THIS post.
    # -----------------------------------------------------

    print(
        f"Best similarity for this post: "
        f"{post_best_similarity:.4f}"
    )

    # -----------------------------------------------------
    # Only consider this social-media post if its
    # strongest detected face passes the threshold.
    # -----------------------------------------------------

    if post_best_similarity < THRESHOLD:

        print(
            f"Rejected: best similarity "
            f"{post_best_similarity:.4f} "
            f"is below threshold "
            f"{THRESHOLD:.2f}."
        )

        continue

    print(
        f"ACCEPTED: face similarity "
        f"{post_best_similarity:.4f} "
        f"meets threshold "
        f"{THRESHOLD:.2f}."
    )

    # -----------------------------------------------------
    # Keep the strongest verified social-media post.
    # -----------------------------------------------------

    if post_best_similarity > candidate_similarity:

        candidate_similarity = (
            post_best_similarity
        )

        candidate = match

        candidate_embedding = (
            post_best_embedding
        )

        with open(
            candidate_path,
            "wb"
        ) as output_file:

            output_file.write(
                post_best_image_response.content
            )

        print(
            "New best verified candidate saved."
        )

  
    

# =========================================================
# 9. Best verified candidate
# =========================================================

print("\n========================================")
print("BEST VERIFIED CANDIDATE")
print("========================================")

if candidate is None:
    print(
        "No social-media candidate could be "
        "verified using face recognition."
    )
    raise SystemExit(0)

print(
    "Title:",
    candidate.get("title")
)

print(
    "Source:",
    candidate.get("source")
)

print(
    "URL:",
    candidate.get("link")
)

print(
    f"Best cosine similarity: "
    f"{candidate_similarity:.4f}"
)

print(
    f"Verified image saved to: "
    f"{candidate_path}"
)


# =========================================================
# 10. Final face verification
# =========================================================

THRESHOLD = 0.60

print("\n========================================")
print("FACE VERIFICATION")
print("========================================")

print(
    f"Cosine similarity: "
    f"{candidate_similarity:.4f}"
)

if candidate_embedding is None:
    raise RuntimeError(
        "No candidate face embedding was stored."
    )

if candidate_similarity < THRESHOLD:

    print(
        "\nRESULT: FACE NOT MATCHED"
    )

    print(
        "The social-media candidate did not "
        "meet the face similarity threshold."
    )

    raise SystemExit(0)


print(
    "\nRESULT: FACE MATCH"
)

print(
    "The social-media candidate "
    "contains a face sufficiently similar "
    "to the input face."
)


# =========================================================
# 11. Upload verified result to blockchain
# =========================================================

print("\n========================================")
print("BLOCKCHAIN VERIFICATION")
print("========================================")

blockchain = LocalBlockchain()

discovered_data = {
    "source": candidate.get("source"),
    "title": candidate.get("title"),
    "url": candidate.get("link"),
    "face_similarity": float(candidate_similarity)
}

block = blockchain.upload(
    discovered_data
)

print("\nBlockchain record created:")

print(
    json.dumps(
        block,
        indent=2
    )
)

verified = blockchain.verify(
    block
)

print(
    "\nBlockchain verification:",
    verified
)

if verified:
    print(
        "RESULT: BLOCKCHAIN VERIFICATION PASSED"
    )
else:
    print(
        "RESULT: BLOCKCHAIN VERIFICATION FAILED"
    )
