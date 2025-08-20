import torch    

def systematic_resample(weights):
    """Performs the systematic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions. A single random
    offset is used to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of weights as floats
    Returns
    -------
    indexes : torch.Tensor
        Tensor of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # Make N subdivisions, and choose positions with a consistent random offset
    # positions = (torch.rand(1) + torch.arange(N)) / N
    positions = (torch.arange(N, device=weights.device) + torch.rand(1, device=weights.device)) / N

    # Initialize indexes array
    indexes = torch.zeros(N, dtype=torch.int64)

    # Calculate the cumulative sum of weights
    cumulative_sum = torch.cumsum(weights, dim=0)

    # Initialize pointers
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


def stratified_resample(weights):
    """Performs the stratified resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions and ensures at least
    one sample is chosen from each division.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of weights as floats
    Returns
    -------
    indexes : torch.Tensor
        Tensor of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # Create an array of equally spaced intervals with a random offset in each
    positions = (torch.arange(N, dtype=torch.float32) + torch.rand(N)) / N

    # Initialize indexes array
    indexes = torch.zeros(N, dtype=torch.int64)

    # Calculate the cumulative sum of weights
    cumulative_sum = torch.cumsum(weights, dim=0)

    # Initialize pointers
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes

def multinomial_resample(weights):
    """Performs multinomial resampling.
    Parameters
    ----------
    weights : torch.Tensor
        Tensor of weights as floats
    Returns
    -------
    indexes : torch.Tensor
        Tensor of indexes into the weights defining the resample.
    """
    N = len(weights)
    indexes = torch.multinomial(weights, num_samples=N, replacement=True)
    return indexes