import base64
import glob
import os
from typing import List

import requests
from qdrant_client import QdrantClient, models

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "colqwen_demo"
EMBEDDING_API_URL = "http://localhost:8888/v1/embeddings"
MODEL_NAME = "qnguyen3/colqwen2.5-v0.2-mlx"
VECTOR_SIZE = 128  # ColQwen/ColPali dimension


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and convert it to a base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Determine mime type based on extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime_type};base64,{encoded_string}"


def get_embeddings(inputs: List[str], model: str = MODEL_NAME) -> List[List[List[float]]]:
    """
    Get multi-vector embeddings from the MLX server.
    Returns a list of embeddings, where each embedding is a list of vectors (List[float]).
    """
    payload = {"input": inputs, "model": model}
    try:
        response = requests.post(EMBEDDING_API_URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to embedding server at {EMBEDDING_API_URL}")
        print("Please ensure the MLX embedding server is running.")
        exit(1)
    except Exception as e:
        print(f"Error calling embedding API: {e}")
        if "response" in locals():
            print(f"Server response: {response.text}")
        exit(1)

    data = response.json()["data"]
    # Sort by index to ensure order matches input
    data.sort(key=lambda x: x["index"])

    # Return raw list of list of floats (no torch conversion needed for Qdrant)
    return [x["embedding"] for x in data]


def cleanup_collection(client: QdrantClient):
    """Delete the Qdrant collection if it exists to ensure a fresh start."""
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting for cleanup...")
        client.delete_collection(COLLECTION_NAME)
        print("Cleanup complete.")


def create_collection(client: QdrantClient):
    """Create the Qdrant collection with MultiVector config."""
    print(f"Creating collection '{COLLECTION_NAME}' with MAX_SIM comparator...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
        ),
    )


def main():
    # 1. Initialize Qdrant Client
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    try:
        cleanup_collection(client)
        create_collection(client)
    except Exception as e:
        print(f"Failed to setup Qdrant collection: {e}")
        print("Please ensure Qdrant is running (run scripts/qdrant-docker.sh)")
        return

    # 2. Prepare Data (Images)
    image_dir = "images"
    if not os.path.exists(image_dir):
        # Fallback to check relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_dir = os.path.join(script_dir, "../images")
        if os.path.exists(possible_dir):
            image_dir = possible_dir

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    if not image_paths:
        print(f"No images found in {image_dir}")
        print("Please add some images to the images/ directory to test indexing.")
        return

    print(f"Found {len(image_paths)} images: {[os.path.basename(p) for p in image_paths]}")

    # 3. Index Images
    print("\nEncoding and indexing images...")

    points = []
    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"Processing {filename}...", end=" ", flush=True)

        # Convert image to base64 for API
        image_uri = encode_image_to_base64(path)

        # Get embedding (returns List[List[float]])
        # We pass one input at a time here for simplicity, but batching is possible
        embeddings = get_embeddings([image_uri])[0]

        # Create point
        points.append(models.PointStruct(id=idx, vector=embeddings, payload={"filename": filename, "path": path}))
        print("Done.")

    # Upsert to Qdrant
    if points:
        print(f"Upserting {len(points)} points to Qdrant...")
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("Indexing complete.")

    # 4. Perform Search
    query_text = "What's on the pink bed?"

    # You can change the query here or take user input
    # query_text = input("\nEnter a search query: ") or "What's on the pink bed?"

    print(f"\nSearching for: '{query_text}'")

    # Embed the query
    query_embedding = get_embeddings([query_text])[0]

    # Search
    search_result = client.query_points(collection_name=COLLECTION_NAME, query=query_embedding, limit=3)

    print("\nSearch Results:")
    for result in search_result.points:
        filename = result.payload.get("filename", "unknown")
        print(f"- {filename} (Score: {result.score:.4f})")


if __name__ == "__main__":
    main()
