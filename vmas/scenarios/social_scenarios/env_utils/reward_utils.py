import torch


def get_root_mean_square(arr: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(torch.square(arr)))


def normalize(arr: torch.Tensor, lims: torch.Tensor) -> torch.Tensor:
    return (arr - lims[:, 0]) / (lims[:, 1] - lims[:, 0])


def mog_reward(
    objectives: torch.Tensor,
    centers: torch.Tensor,
    sigmas: torch.Tensor,
    lims: torch.Tensor,
    A: float = 1.0,
) -> torch.Tensor:
    """Use a Multi-Objective Gaussian function to calculate the reward.
    Args:
      objectives (torch.Tensor): The objectives signals.
        it is a tensor of shape (num_envs, num_metrics).
      centers (torch.Tensor): The center (objective value) for the Gaussian kernel.
        it is a tensor of shape (num_metrics,).
      sigmas (torch.Tensor): The sigma value for the Gaussian kernel.
        it a tensor of shape (num_metrics,).
      lims (torch.Tensor): The limits for the normalization.
        it is a tensor of shape (num_metrics, 2).
    Returns:
      torch.Tensor: The calculated reward.
        it is a tensor of shape (num_envs,)."""

    # calculate the distance between the objectives and the centers
    #
    # The distance is calculated using the Euclidean distance
    # between the objectives and the centers.

    distances = torch.sqrt(torch.square(objectives - centers))
    # normalize and clip the distances (between 0 and 1) using the limits
    distances_nor = torch.clip(normalize(distances, lims), 0.0, 1.0)

    # calculate the reward using the Gaussian kernel
    h = torch.pow(distances_nor, 2) / (2 * sigmas)
    return A * torch.exp(-torch.sum(h, dim=1))  # sum over metrics [num_envs,]


def mocg_reward(
    objectives: torch.Tensor,
    centers: torch.Tensor,
    sigmas: torch.Tensor,
    lims: torch.Tensor,
    A: float = 1.0,
) -> torch.Tensor:
    """Use a Multi-Objective cascaded Gaussian function to calculate the reward.
    Args:
      objectives (torch.Tensor): The objectives signals.
        it is a tensor of shape (num_envs, num_metrics).
      centers (torch.Tensor): The center (objective value) for the Gaussian kernel.
        it is a tensor of shape (num_metrics,).
      sigmas (torch.Tensor): The sigma value for the Gaussian kernel.
        it a tensor of shape (num_metrics,).
      lims (torch.Tensor): The limits for the normalization.
        it is a tensor of shape (num_metrics, 2).
      A (float): The amplitude of the reward.
    Returns:
      torch.Tensor: The calculated reward.
        it is a tensor of shape (num_envs,)."""

    semi_rewards = objectives.clone()
    new_centers = centers.clone()
    new_sigmas = sigmas.clone()
    new_lims = lims.clone()
    while semi_rewards.shape[1] > 1:
        for i in range(0, semi_rewards.shape[1], 2):
            # process the pairs of semi-rewards using the MOG function in pairs
            semi_rewards[:, i // 2] = (
                mog_reward(
                    semi_rewards[:, i : i + 2],
                    new_centers[:, i : i + 2],
                    new_sigmas[:, i : i + 2],
                    new_lims[:, i : i + 2],
                    A,
                )
                - A
            )  # subtract A to avoid double counting
            new_centers[:, i // 2] = 0.0
            new_sigmas[:, i // 2] = 1.0
            new_lims[:, i // 2] = torch.tensor([0.0, 1.0])

        # if there is an odd number of semi-rewards, process the last one
        if semi_rewards.shape[1] % 2 == 1:
            semi_rewards[:, -1] = semi_rewards[:, -1]
            new_centers[:, -1] = new_centers[:, -1]
            new_sigmas[:, -1] = new_sigmas[:, -1]
            new_lims[:, -1] = new_lims[:, -1]

        # reduce the size of the semi_rewards, new_centers, new_sigmas, and new_lims tensors
        semi_rewards = semi_rewards[:, : (semi_rewards.shape[1] + 1) // 2]
        new_centers = new_centers[:, : (new_centers.shape[1] + 1) // 2]
        new_sigmas = new_sigmas[:, : (new_sigmas.shape[1] + 1) // 2]
        new_lims = new_lims[:, : (new_lims.shape[1] + 1) // 2]

    return semi_rewards[:, 0]
