import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ---------------------------------------------------------
# 1. Initialize InsightFace
# ---------------------------------------------------------

print("Initializing InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("InsightFace initialized successfully.\n")


# ---------------------------------------------------------
# 2. Image paths
# ---------------------------------------------------------

# Same person
same_person_1 = (
    r"dataset\lfw-deepfunneled\lfw-deepfunneled"
    r"\Aaron_Peirsol\Aaron_Peirsol_0001.jpg"
)

same_person_2 = (
    r"dataset\lfw-deepfunneled\lfw-deepfunneled"
    r"\Aaron_Peirsol\Aaron_Peirsol_0002.jpg"
)

# Different person
different_person = (
    r"dataset\lfw-deepfunneled\lfw-deepfunneled"
    r"\Aaron_Sorkin\Aaron_Sorkin_0001.jpg"
)


# ---------------------------------------------------------
# 3. Helper function to get a face embedding
# ---------------------------------------------------------

def get_embedding(image_path):
    """Detect the first face in an image and return its embedding."""

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    faces = app.get(image)

    if len(faces) == 0:
        raise RuntimeError(
            f"No face detected in: {image_path}"
        )

    return faces[0].embedding


# ---------------------------------------------------------
# 4. Cosine similarity function
# ---------------------------------------------------------

def cosine_similarity(embedding_a, embedding_b):
    """Calculate cosine similarity between two face embeddings."""

    denominator = (
        np.linalg.norm(embedding_a)
        * np.linalg.norm(embedding_b)
    )

    if denominator == 0:
        raise ValueError("Cannot calculate similarity for a zero embedding.")

    return np.dot(embedding_a, embedding_b) / denominator


# ---------------------------------------------------------
# 5. Generate embeddings
# ---------------------------------------------------------

print("Generating face embeddings...\n")

embedding_same_1 = get_embedding(same_person_1)
embedding_same_2 = get_embedding(same_person_2)
embedding_different = get_embedding(different_person)


# ---------------------------------------------------------
# 6. Compare faces
# ---------------------------------------------------------

same_similarity = cosine_similarity(
    embedding_same_1,
    embedding_same_2
)

different_similarity = cosine_similarity(
    embedding_same_1,
    embedding_different
)


# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------

print("========================================")
print("FACE VERIFICATION RESULTS")
print("========================================")

print(
    f"Same person similarity:      {same_similarity:.4f}"
)

print(
    f"Different person similarity: {different_similarity:.4f}"
)

print("========================================")


# ---------------------------------------------------------
# 8. Verification
# ---------------------------------------------------------

if same_similarity > different_similarity:
    print("\nPASS: Same-person similarity is higher.")
else:
    print("\nWARNING: Verification test did not behave as expected.")