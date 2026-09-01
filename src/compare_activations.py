import torch
import torch.nn.functional as F


def compare_act(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    cosine_similarity = F.cosine_similarity(x, y, dim=-1)
    euclidean_distance = torch.linalg.vector_norm(x - y, dim=-1)

    return cosine_similarity, euclidean_distance


positive_activations = torch.load(DATA_DIR / "difference_in_means" / "pos_act.pt")
negative_activations = torch.load(DATA_DIR / "difference_in_means" / "neg_act.pt")
print(positive_activations.size(), negative_activations.size())
cosine_similarity, euclidean_distance = compare_act(
    positive_activations, negative_activations
)
print(cosine_similarity.size(), euclidean_distance.size())
