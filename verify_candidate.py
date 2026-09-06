import cv2
import numpy as np
from insightface.app import FaceAnalysis

print("Initializing InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

def get_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(image)

    if not faces:
        raise ValueError(f"No face detected: {image_path}")

    return faces[0].embedding

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


input_image = (
    "dataset\\lfw-deepfunneled\\lfw-deepfunneled"
    "\\Aaron_Peirsol\\Aaron_Peirsol_0001.jpg"
)

candidate_image = "candidate_social.jpg"

print("\nLoading candidate...")
candidate_embedding = get_embedding(candidate_image)

print("\n========================================")
print("AARON PEIRSOL REFERENCE COMPARISON")
print("========================================")

scores = []

for i in range(1, 5):

    reference = (
        "dataset\\lfw-deepfunneled\\lfw-deepfunneled"
        "\\Aaron_Peirsol"
        f"\\Aaron_Peirsol_{i:04d}.jpg"
    )

    reference_embedding = get_embedding(reference)

    score = cosine_similarity(
        candidate_embedding,
        reference_embedding
    )

    scores.append(score)

    print(
        f"Aaron_Peirsol_{i:04d}.jpg: "
        f"{score:.4f}"
    )

print("\n========================================")
print("RESULT")
print("========================================")

print(f"Highest similarity: {max(scores):.4f}")
print(f"Average similarity: {np.mean(scores):.4f}")
print(f"Lowest similarity: {min(scores):.4f}")