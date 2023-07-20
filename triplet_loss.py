import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_distance_matrix(
        embeddings: torch.Tensor,  #  [B, E]
    ):
    B = embeddings.size(0)
    dot_product = embeddings @ embeddings.T  # [B, B]
    squared_norm = torch.diag(dot_product) # [B]
    distances = squared_norm.view(1, B) - 2.0 * dot_product + squared_norm.view(B, 1)  # [B, B]
    return torch.sqrt(nn.functional.relu(distances) + 1e-16)  # [B, B]
  
def get_positive_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    indices_equal = torch.eye(B, dtype=torch.bool).cuda()  # [B, B]
    return labels_equal & ~indices_equal  # [B, B]

def get_negative_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):
    B = labels.size(0)
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    indices_equal = torch.eye(B, dtype=torch.bool).cuda()  # [B, B]
    return ~labels_equal & ~indices_equal  # [B, B]

def get_triplet_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
    ):

    B = labels.size(0)

    # Make sure that i != j != k
    indices_equal = torch.eye(B, dtype=torch.bool).cuda()  # [B, B]
    indices_not_equal = ~indices_equal  # [B, B]
    i_not_equal_j = indices_not_equal.view(B, B, 1)  # [B, B, 1]
    i_not_equal_k = indices_not_equal.view(B, 1, B)  # [B, 1, B]
    j_not_equal_k = indices_not_equal.view(1, B, B)  # [1, B, B]
    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k  # [B, B, B]

    # Make sure that labels[i] == labels[j] but labels[i] != labels[k]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1)  # [B, B, 1]
    i_equal_k = labels_equal.view(B, 1, B)  # [B, 1, B]
    valid_labels = i_equal_j & ~i_equal_k  # [B, B, B]

    return distinct_indices & valid_labels  # [B, B, B]

def test_get_distance_matrix(device_for_tests):
    embeddings = torch.FloatTensor(
        [[1, 1], 
        [7, 7], 
        [1, 1]], 
    ).to(device=device_for_tests)
    distance_matrix = get_distance_matrix(embeddings)
    assert torch.allclose(
        torch.diag(distance_matrix), 
        torch.zeros(3, device=device_for_tests)
    )
    assert torch.allclose(distance_matrix, distance_matrix.T)
    assert distance_matrix[0, 2] < distance_matrix[0, 1]


def test_get_positive_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1])
    pos_mask = get_positive_mask(labels, device_for_tests)
    assert pos_mask[0, 3]
    assert not pos_mask[0, 1]
    assert not pos_mask[0, 0] and not pos_mask[1, 1]


def test_get_negative_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1])
    neg_mask = get_negative_mask(labels, device_for_tests)
    assert not neg_mask[0, 3]
    assert neg_mask[0, 1]
    assert not neg_mask[0, 0] and not neg_mask[1, 1]


def test_get_triplet_mask(device_for_tests):
    labels = torch.LongTensor([1, 2, 3, 1, 3])
    mask = get_triplet_mask(labels, device_for_tests)
    assert mask[0, 3, 2]
    assert mask[2, 4, 1]
    assert mask[4, 2, 0]
    assert not mask[0, 0, 0]
    assert not mask[0, 3, 3]
    assert not mask[0, 0, 4]

class TripletLoss(nn.Module):
    
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet
        self.resnet.fc = nn.Identity()
        self.embeddings = nn.Linear(512, 10).cuda()
        
    def forward(
            self, 
            inputs: torch.Tensor,  # [B, C, H, W]
            labels: torch.Tensor  # [B]
        ):
        B = labels.size(0)
        embeddings = self.embeddings(inputs)  # [B, E]
        distance_matrix = get_distance_matrix(embeddings)  # [B, B]
        with torch.no_grad():
            mask_pos = get_positive_mask(labels, device)  # [B, B]
            mask_neg = get_negative_mask(labels, device)  # [B, B]
            triplet_mask = get_triplet_mask(labels, device)  # [B, B, B]
            unmasked_triplets = torch.sum(triplet_mask)  # [1]
            mu_pos = torch.mean(distance_matrix[mask_pos])  # [1]
            mu_neg = torch.mean(distance_matrix[mask_neg])  # [1]
            mu = mu_neg - mu_pos  # [1]
        
        distance_i_j = distance_matrix.view(B, B, 1)  # [B, B, 1]
        distance_i_k = distance_matrix.view(B, 1, B)  # [B, 1, B]
        triplet_loss_unmasked = distance_i_k - distance_i_j   # [B, B, B]
        triplet_loss_unmasked = triplet_loss_unmasked[triplet_mask] # [valid_triplets]
        hardest_triplets = triplet_loss_unmasked < max(mu, 0)  # [valid_triplets]
        triplet_loss = triplet_loss_unmasked[hardest_triplets]  # [valid_triplets_after_mask]
        triplet_loss = nn.functional.relu(triplet_loss)  # [valid_triplets_after_mask]

        loss = triplet_loss.mean()
        """
        logs = {
            'positive_pairs': torch.sum(mask_pos).cpu().detach().item(),
            'negative_pairs': torch.sum(mask_neg).cpu().detach().item(),
            'mu_neg': mu_neg.cpu().detach().item(),
            'mu_pos': mu_pos.cpu().detach().item(),
            'valid_triplets': unmasked_triplets.cpu().detach().item(),
            'valid_triplets_after_mask': triplet_loss.size(0),
            'triplet_loss': triplet_loss.mean().cpu().detach().item()
        }
        """
        return loss