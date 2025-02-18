"""This module contains basic computing functions."""

import numba as nb
import numpy as np

@nb.njit(parallel=True)
def compute_distance_matrix(particle_positions, box_length):
    """
    Compute the distance matrix between particles with periodic boundary conditions.

    Parameters:
        particle_positions (np.ndarray): (N, 3) array containing particle positions.
        box_length (np.ndarray): (3,) array containing box dimensions.

    Returns:
        np.ndarray: Distance matrix between particles.
    """
    num_molecule = particle_positions.shape[0]
    r_matrix = np.empty((num_molecule, num_molecule), dtype=np.float64)
    
    for i in nb.prange(num_molecule):
        for j in range(i + 1, num_molecule):
            # r vector
            r_vec = particle_positions[i, :] - particle_positions[j, :]

            # Apply periodic boundary conditions
            r_vec -= box_length * np.round(r_vec / box_length)

            # Compute the distance
            r = np.sqrt(np.sum(r_vec ** 2))
            r_matrix[i, j] = r
            r_matrix[j, i] = r  # Symmetric matrix

        # Set diagonal elements to NaN
        r_matrix[i, i] = np.nan

    return r_matrix


def compute_local_entropy(particle_positions, cutoff, sigma, use_local_density, box_length, compute_average, cutoff2=0):
    """
    Compute the local entropy of particles in a system.

    Parameters:
        particle_positions (np.ndarray): (N, 3) array containing particle positions.
        cutoff (float): Cutoff distance for the entropy calculation.
        sigma (float): Broadening parameter.
        use_local_density (bool): Whether to use local density for the entropy calculation.
        box_length (np.ndarray): (3,) array containing box dimensions.
        compute_average (bool): Whether to compute the spatially averaged entropy.
        cutoff2 (float): Cutoff distance for spatial averaging.

    Returns:
        np.ndarray: Local entropy values for each particle.
    """
    # Number of particles
    num_particle = particle_positions.shape[0]
    
    # Overall particle density:
    volume = box_length[0] * box_length[1] * box_length[2]
    global_rho = num_particle / volume

    # Create output array for local entropy values
    local_entropy = np.empty(num_particle)

    distance_matrix = compute_distance_matrix(particle_positions=particle_positions, box_length=box_length)

    # Number of bins used for integration:
    nbins = int(cutoff / sigma) + 1

    # Table of r values at which the integrand will be computed:
    r = np.linspace(0.0, cutoff, num=nbins)
    rsq = r**2

    # Precompute normalization factor of g_m(r) function:
    prefactor = rsq * (4 * np.pi * global_rho * np.sqrt(2 * np.pi * sigma**2))
    prefactor[0] = prefactor[1] # Avoid division by zero at r=0.

    # Iterate over input particles:
    for particle_index in range(num_particle):
        # Get distances r_ij of neighbors within the cutoff range.
        r_ij = distance_matrix[particle_index, distance_matrix[particle_index, :] < cutoff] 

        # Compute differences (r - r_ji) for all {r} and all {r_ij} as a matrix.
        r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)

        # Compute g_m(r):
        g_m = np.sum(np.exp(-r_diff**2 / (2.0*sigma**2)), axis=0) / prefactor

        # Estimate local atomic density by counting the number of neighbors within the
        # spherical cutoff region:
        if use_local_density:
            local_volume = 4/3 * np.pi * cutoff**3
            rho = len(r_ij) / local_volume
            g_m *= global_rho / rho
        else:
            rho = global_rho

        # Compute integrand:
        valid_g_m = np.maximum(g_m, 1e-10) 
        integrand = np.where(g_m >= 1e-10, (g_m * np.log(valid_g_m) - g_m + 1.0) * rsq, rsq)

        # Integrate from 0 to cutoff distance:
        local_entropy[particle_index] = -2.0 * np.pi * rho * np.trapz(integrand, r)

    # If requested, perform the spatial averaging of the local entropy value 
    if compute_average:
        local_entropy_avg = np.empty(num_particle)
        for particle_index in range(num_particle):
            idx = distance_matrix[particle_index, :] < (cutoff2)
            local_entropy_avg[particle_index] = (np.sum(local_entropy[idx]) + local_entropy[particle_index]) / (np.sum(idx) + 1)
        return local_entropy_avg
    else:
        return local_entropy

def compute_local_fraction(mol_positions, num_mol_1, box_length, cutoff):
    """
    Computes the local fraction of molecules of type 1 within a cutoff distance.
    
    Parameters:
        mol_positions (np.ndarray): (N, 3) array containing molecular positions.
        num_mol_1 (int): Number of molecules of type 1.
        box_length (np.ndarray): (3,) array containing box dimensions.
        cutoff (float): Cutoff distance for counting neighbors.
    
    Returns:
        np.ndarray: Local fraction of type 1 molecules around each molecule.
    """
    num_mol = mol_positions.shape[0]
    local_fraction = np.zeros(num_mol)

    distance_matrix = compute_distance_matrix(particle_positions=mol_positions, box_length=box_length)
    np.fill_diagonal(distance_matrix, 0)

    within_cutoff = distance_matrix < cutoff
    
    count_neighbors = np.sum(within_cutoff, axis=1)
    count_num_mol_1 = np.sum(within_cutoff[:, :num_mol_1], axis=1)

    local_fraction = count_num_mol_1 / count_neighbors

    return local_fraction