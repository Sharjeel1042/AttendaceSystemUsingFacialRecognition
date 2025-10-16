import cv2
import numpy as np

def get_embedding(face, embedder):
    """
    Generates a 128-d face embedding using the FaceNet model.

    Args:
        face (numpy.ndarray): The extracted face image (160x160).
        embedder (FaceNet): The loaded FaceNet model.

    Returns:
        numpy.ndarray: The 128-d feature vector (embedding) for the face.
    """
    # FaceNet model expects RGB images
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Add batch dimension
    face = np.expand_dims(face, axis=0)
    # Get the embedding
    embedding = embedder.embeddings(face)[0]
    return embedding

