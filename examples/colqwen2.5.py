import base64
import glob
import os
from typing import List

import requests
import torch


def late_interaction_score(
    query_emb: torch.Tensor,  # (n_queries, n_query_tokens, dim)   e.g. (2, 6, 128)
    doc_emb: torch.Tensor,  # (n_docs,    n_doc_patches,  dim)   e.g. (1, 397, 128)
) -> torch.Tensor:
    """
    Compute ColQwen / ColBERT-style late interaction score (sum of MaxSim operations).
    Returns: tensor of shape (n_queries, n_docs) with similarity scores.
    Higher = more similar.
    """
    # Ensure same dtype & device
    query_emb = query_emb.to(doc_emb.dtype).to(doc_emb.device)
    # Expand dims so we can broadcast:
    # query: (n_q, n_qtok, 1,    dim)
    # doc:   (1,   1,      n_doc, n_patch, dim)  wait — better way below

    # Core operation: dot product between every query token and every doc patch
    # (n_queries, n_q_tokens, n_docs, n_patches)
    dots = torch.einsum(
        "qtd,pcd->qtpc",
        query_emb,
        doc_emb,  # q=query batch, t=token, p=doc batch, c=patch, d=dim  # q t d  # p c d
    )

    # For each query token → max similarity to any patch in that doc
    # → shape (n_queries, n_q_tokens, n_docs)
    max_sim_per_token = dots.max(dim=-1).values  # max over patches (dim=-1)

    # Sum over query tokens → final score per query-doc pair
    # shape (n_queries, n_docs)
    scores = max_sim_per_token.sum(dim=1)

    return scores


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    # Determine mime type based on extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime_type};base64,{encoded_string}"


def get_embeddings(inputs: List[str], model: str = "qnguyen3/colqwen2.5-v0.2-mlx") -> torch.Tensor:
    url = "http://localhost:8888/v1/embeddings"
    payload = {"input": inputs, "model": model}
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.text}")
        response.raise_for_status()

    data = response.json()["data"]
    # Sort by index to ensure order
    data.sort(key=lambda x: x["index"])
    embeddings = [x["embedding"] for x in data]
    return torch.tensor(embeddings)


def main():
    # query = "What percentage of the data is books?"
    query = "What's on the pink bed?"
    # query = "A photo of 2 cats on pink bedsheets, with remote control next to the cats."
    image_dir = "images"

    if not os.path.exists(image_dir):
        # Fallback to check relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_dir = os.path.join(script_dir, "../images")
        if os.path.exists(possible_dir):
            image_dir = possible_dir

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    # Filter for image extensions
    image_paths = [p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images: {[os.path.basename(p) for p in image_paths]}")

    # 1. Get Query Embedding
    print(f"Encoding query: {query}")
    query_emb = get_embeddings([query])  # Shape (1, tokens, dim)

    # 2. Get Image Embeddings and calculate scores one by one
    print("\nEncoding images and calculating scores individually...")
    score_list = []

    for path in image_paths:
        print(f"Processing {os.path.basename(path)}...", end="", flush=True)
        img_b64 = encode_image_to_base64(path)
        doc_emb = get_embeddings([img_b64])  # Shape (1, patches, dim)

        # Calculate Score for this pair
        score = late_interaction_score(query_emb, doc_emb)
        final_score = score.item()

        print(f" Score: {final_score:.4f}")
        score_list.append((path, final_score))

    # Sort by score desc
    score_list.sort(key=lambda x: x[1], reverse=True)

    print("\nFinal Results (Sorted):")
    for path, score in score_list:
        print(f"{score:.4f}  {os.path.basename(path)}")


if __name__ == "__main__":
    main()
