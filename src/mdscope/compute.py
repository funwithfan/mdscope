"""This module contains basic computing functions."""

import numba as nb
import numpy as np

# Define unit conversion
kcal_mol2j = 6.9477e-21
A2m = 1e-10
e2c = 1.60217663e-19
a_fs2m_s = 1e5
g_mol2kg = 1.6611e-27

# Define basic parameters
kB = 1.380649E-23
eps_0 = 8.85418782e-12      # F/m, dielectric permittivity
C = 1 / (4 * np.pi * eps_0) # energy-conversion constant
dielectric = 1              # dielectric constant

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


@nb.njit()
def get_mol_ke(mol1_velocities, mol2_velocities, atom_type_single_mol1, atom_type_single_mol2, mass):
    """
    Compute the kinetic energy of each molecule.
    """
    num_mol1 = mol1_velocities.shape[0]
    num_mol2 = mol2_velocities.shape[0]
    num_mol = num_mol1 + num_mol2

    mol_ke = np.zeros(num_mol)

    for i in range(num_mol1):
        for j in range(mol1_velocities.shape[1]):
            mol_ke[i] += mass[atom_type_single_mol1[j]] * np.sum(mol1_velocities[i,j,:]**2)
    
    for i in range(num_mol2):
        for j in range(mol2_velocities.shape[1]):
            mol_ke[i+num_mol1] += mass[atom_type_single_mol2[j]] * np.sum(mol2_velocities[i,j,:]**2)

    mol_ke *= 0.5 * g_mol2kg * a_fs2m_s**2 / kcal_mol2j # unit: kcal_mol
    return mol_ke

@nb.njit(parallel=True)
def get_com_velocity(mol_velocities, mol_mass):
    """
    Compute the center of mass velocity for each molecule.
    """
    num_mol = mol_velocities.shape[0]

    com_velocities = np.zeros((num_mol, 3))

    m = np.sum(mol_mass)

    for i in nb.prange(num_mol):
        com_velocities[i] = np.sum(mol_velocities[i] * mol_mass[:, np.newaxis], axis=0) / m
    
    return com_velocities

@nb.njit()
def get_com_velocity_vectorized(mol_velocities, mol_mass):
    """
    向量化计算多个分子中心的质量速度 (Center-of-Mass Velocity)。

    Parameters:
    mol_velocities (ndarray): 形状为 (num_mol, num_atoms_per_mol, 3)，表示每个分子的原子速度。
    mol_mass (ndarray): 形状为 (num_atoms_per_mol,)，表示每个分子的原子质量。

    Returns:
    ndarray: 形状为 (num_mol, 3)，表示每个分子的中心质量速度。
    """
    # 计算总质量（假设所有分子的原子质量相同）
    total_mass = np.sum(mol_mass)

    # 广播原子质量以匹配速度形状 (num_mol, num_atoms_per_mol, 3)
    weighted_velocities = mol_velocities * mol_mass[:, np.newaxis]

    # 对每个分子的加权速度求和，并除以总质量
    com_velocities = np.sum(weighted_velocities, axis=1) / total_mass

    return com_velocities


@nb.njit()
def inter_molecular_pe(mol1_position, mol2_position, mol1_type, mol2_type, eps, sig, q, box_length, cutoff):
    """
    Compute the intermolecular potential energy between two molecules.
    """
    vdw = 0
    coul = 0

    num_atom_mol1 = mol1_position.shape[0]
    num_atom_mol2 = mol2_position.shape[0]

    for i in range(num_atom_mol1):
        for j in range(num_atom_mol2):
            r_vec = mol1_position[i, :] - mol2_position[j, :]

            # Apply periodic boundary conditions along each dimension
            r_vec -= box_length * np.round(r_vec / box_length)

            # Compute the distance
            r = np.linalg.norm(r_vec)

            if r > cutoff:
                return 0.0
            else:
            # if r < cutoff:
                # Get the atom types for the interaction
                type1 = mol1_type[i]
                type2 = mol2_type[j]

                # Lennard-Jones parameters
                epsilon = eps[type1, type2]
                sigma = sig[type1, type2]

                # Lennard-Jones potential
                r6 = (sigma / r)**6
                r12 = r6**2
                vdw += 4 * epsilon * (r12 - r6)

                # Coulombic potential
                q1 = q[type1]
                q2 = q[type2] 
                coul += (C * q1 * q2) / (dielectric * r)

    pe = vdw + coul * e2c * e2c / A2m / kcal_mol2j  # unit: kcal/mol
    return pe


@nb.njit(parallel=True)
def get_pe_matrix(mol1_positions, mol2_positions, atom_type_single_mol1, atom_type_single_mol2, eps, sig, q, box_length, cutoff):
    """
    Compute the potential energy matrix between all pairs of molecules.
    """
    num_mol1 = mol1_positions.shape[0]
    num_mol2 = mol2_positions.shape[0]
    num_mol = num_mol1 + num_mol2

    pe_matrix = np.zeros((num_mol, num_mol))

    for i in nb.prange(num_mol):  # Parallelized loop
        if i < num_mol1:
            mol_i_position = mol1_positions[i]
            mol_i_atom_type = atom_type_single_mol1
        else:
            mol_i_position = mol2_positions[i - num_mol1]
            mol_i_atom_type = atom_type_single_mol2

        for j in range(i + 1, num_mol):
            if j < num_mol1:
                mol_j_position = mol1_positions[j]
                mol_j_atom_type = atom_type_single_mol1
            else:
                mol_j_position = mol2_positions[j - num_mol1]
                mol_j_atom_type = atom_type_single_mol2

            pe_matrix[i, j] = inter_molecular_pe(mol1_postion=mol_i_position,
                                                 mol2_postion=mol_j_position,
                                                 mol1_type=mol_i_atom_type,
                                                 mol2_type=mol_j_atom_type,
                                                 eps=eps,
                                                 sig=sig,
                                                 q=q,
                                                 box_length=box_length,
                                                 cutoff=cutoff)
            pe_matrix[j, i] = pe_matrix[i, j]  # Enforce symmetry

    return pe_matrix


@nb.njit()
def relative_ke(mol1_velocity, mol2_velocity, mol1_mass, mol2_mass):
    """
    Compute the relative kinetic energy between two molecules.
    """
    # Total mass of each molecule
    m1 = np.sum(mol1_mass)
    m2 = np.sum(mol2_mass)

    # Reduced mass
    miu = (m1 * m2) / (m1 + m2)

    # Center of mass velocity for each molecule
    mol1_com_velocity = np.sum(mol1_velocity * mol1_mass[:, np.newaxis], axis=0) / m1
    mol2_com_velocity = np.sum(mol2_velocity * mol2_mass[:, np.newaxis], axis=0) / m2

    # Relative velocity
    relative_velocity = mol1_com_velocity - mol2_com_velocity

    # Relative kinetic energy
    ke = 0.5 * miu * np.dot(relative_velocity, relative_velocity) * g_mol2kg * a_fs2m_s * a_fs2m_s / kcal_mol2j # unit: kcal_mol

    return ke

@nb.njit(parallel=True, fastmath=True)
def hill_criterion(mol1_positions, mol2_positions, mol1_velocities, mol2_velocities,
                   atom_type_single_mol1, atom_type_single_mol2, eps, sig, q, mass,
                   box_length, cutoff):
    num_mol1 = mol1_positions.shape[0]
    num_mol2 = mol2_positions.shape[0]
    num_mol = num_mol1 + num_mol2
    # print(num_mol1, num_mol2)

    # Initialize adjacency matrix and potential energy
    adj_matrix = np.zeros((num_mol, num_mol), dtype=np.bool_)
    pe_matrix = np.zeros((num_mol, num_mol), dtype=np.float64)

    for i in nb.prange(num_mol):  # Parallelized loop
        if i < num_mol1:
            mol_i_atom_type = atom_type_single_mol1
            mol_i_position = mol1_positions[i]
            mol_i_velocity = mol1_velocities[i]
        else:
            mol_i_atom_type = atom_type_single_mol2
            mol_i_position = mol2_positions[i - num_mol1]
            mol_i_velocity = mol2_velocities[i - num_mol1]

        for j in range(i + 1, num_mol):  # Compute upper triangle only
            if j < num_mol1:
                mol_j_atom_type = atom_type_single_mol1
                mol_j_position = mol1_positions[j]
                mol_j_velocity = mol1_velocities[j]
            else:
                mol_j_atom_type = atom_type_single_mol2
                mol_j_position = mol2_positions[j - num_mol1]
                mol_j_velocity = mol2_velocities[j - num_mol1]

            # Compute pair potential energy
            pe = inter_molecular_pe(mol1_position=mol_i_position,
                                    mol2_position=mol_j_position,
                                    mol1_type=mol_i_atom_type,
                                    mol2_type=mol_j_atom_type,
                                    eps=eps,
                                    sig=sig,
                                    q=q,
                                    box_length=box_length,
                                    cutoff=cutoff)

            # potential energy
            pe_matrix[i, j] = pe
            pe_matrix[j, i] = pe

            # Only consider pairs with attractive interaction (PE < 0)
            if pe < 0:
                # Compute relative kinetic energy
                ke = relative_ke(mol1_velocity=mol_i_velocity,
                                 mol2_velocity=mol_j_velocity,
                                 mol1_mass=mass[mol_i_atom_type],
                                 mol2_mass=mass[mol_j_atom_type])

                # Check Hill criterion
                if pe + ke < 0:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Enforce symmetry

    return adj_matrix, pe_matrix

   
@nb.njit(parallel=True, fastmath=True)
def hill_criterion2(mol1_positions, mol2_positions, mol1_velocities, mol2_velocities,
                   atom_type_single_mol1, atom_type_single_mol2, eps, sig, q, mass,
                   box_length, cutoff):
    num_mol1 = mol1_positions.shape[0]
    num_mol2 = mol2_positions.shape[0]
    num_mol = num_mol1 + num_mol2

    # Total mass of each molecule
    m1 = np.sum(mass[atom_type_single_mol1])
    m2 = np.sum(mass[atom_type_single_mol2])

    mol1_com_velocities = get_com_velocity_vectorized(mol1_velocities, mass[atom_type_single_mol1])
    mol2_com_velocities = get_com_velocity_vectorized(mol2_velocities, mass[atom_type_single_mol2])

    # Initialize adjacency matrix and potential energy
    adj_matrix = np.zeros((num_mol, num_mol), dtype=np.bool_)
    pe_matrix = np.zeros((num_mol, num_mol), dtype=np.float64)

    for idx in nb.prange(num_mol * (num_mol - 1) // 2):  # Parallelized loop for the upper triangle
        i = int(num_mol - 2 - int(np.sqrt(-8*idx + 4*num_mol*(num_mol-1)-7)/2.0 - 0.5))
        j = int(idx + i + 1 - num_mol*(num_mol-1)/2 + (num_mol-i)*((num_mol-i)-1)/2)


        if i < num_mol1:
            mol_i_atom_type = atom_type_single_mol1
            mol_i_position = mol1_positions[i]
            mol_i_com_velocity = mol1_com_velocities[i]
            mass_i = m1
        else:
            mol_i_atom_type = atom_type_single_mol2
            mol_i_position = mol2_positions[i - num_mol1]
            mol_i_com_velocity = mol2_com_velocities[i - num_mol1]
            mass_i = m2

        
        if j < num_mol1:
            mol_j_atom_type = atom_type_single_mol1
            mol_j_position = mol1_positions[j]
            mol_j_com_velocity = mol1_com_velocities[j]
            mass_j = m1
        else:
            mol_j_atom_type = atom_type_single_mol2
            mol_j_position = mol2_positions[j - num_mol1]
            mol_j_com_velocity = mol2_com_velocities[j - num_mol1]
            mass_j = m2


        # Compute pair potential energy
        pe = inter_molecular_pe(mol1_position=mol_i_position,
                                mol2_position=mol_j_position,
                                mol1_type=mol_i_atom_type,
                                mol2_type=mol_j_atom_type,
                                eps=eps,
                                sig=sig,
                                q=q,
                                box_length=box_length,
                                cutoff=cutoff)

        # potential energy
        pe_matrix[i, j] = pe
        pe_matrix[j, i] = pe

        # Only consider pairs with attractive interaction (PE < 0)
        if pe < 0:
            # Compute relative kinetic energy
            relative_velocity = mol_i_com_velocity - mol_j_com_velocity # Relative velocity
            miu = (mass_i * mass_j) / (mass_i + mass_j) # Reduced mass
            ke = 0.5 * miu * np.dot(relative_velocity, relative_velocity) * g_mol2kg * a_fs2m_s * a_fs2m_s / kcal_mol2j # Relative kinetic energy, unit: kcal_mol

            # Check Hill criterion
            if pe + ke < 0:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Enforce symmetry

    return adj_matrix, pe_matrix

@nb.njit(parallel=True, fastmath=True)
def hill_criterion3(mol1_positions, mol2_positions, mol1_velocities, mol2_velocities,
                   atom_type_single_mol1, atom_type_single_mol2, eps, sig, q, mass,
                   box_length, cutoff):
    num_mol1 = mol1_positions.shape[0]
    num_mol2 = mol2_positions.shape[0]
    num_mol = num_mol1 + num_mol2

    # Initialize adjacency matrix and potential energy
    adj_matrix = np.zeros((num_mol, num_mol), dtype=np.bool_)
    pe_matrix = np.zeros((num_mol, num_mol), dtype=np.float64)

    for idx in nb.prange(num_mol * (num_mol - 1) // 2):  # Parallelized loop for the upper triangle

        i = int(num_mol - 2 - int(np.sqrt(-8*idx + 4*num_mol*(num_mol-1)-7)/2.0 - 0.5))
        j = int(idx + i + 1 - num_mol*(num_mol-1)/2 + (num_mol-i)*((num_mol-i)-1)/2)

        if i < num_mol1:
            mol_i_atom_type = atom_type_single_mol1
            mol_i_position = mol1_positions[i]
            mol_i_velocity = mol1_velocities[i]
        else:
            mol_i_atom_type = atom_type_single_mol2
            mol_i_position = mol2_positions[i - num_mol1]
            mol_i_velocity = mol2_velocities[i - num_mol1]

   
        if j < num_mol1:
            mol_j_atom_type = atom_type_single_mol1
            mol_j_position = mol1_positions[j]
            mol_j_velocity = mol1_velocities[j]
        else:
            mol_j_atom_type = atom_type_single_mol2
            mol_j_position = mol2_positions[j - num_mol1]
            mol_j_velocity = mol2_velocities[j - num_mol1]

        # Compute pair potential energy
        pe = inter_molecular_pe(mol1_position=mol_i_position,
                                mol2_position=mol_j_position,
                                mol1_type=mol_i_atom_type,
                                mol2_type=mol_j_atom_type,
                                eps=eps,
                                sig=sig,
                                q=q,
                                box_length=box_length,
                                cutoff=cutoff)

        # potential energy
        pe_matrix[i, j] = pe
        pe_matrix[j, i] = pe

        # Only consider pairs with attractive interaction (PE < 0)
        if pe < 0:
            # Compute relative kinetic energy
            ke = relative_ke(mol1_velocity=mol_i_velocity,
                                mol2_velocity=mol_j_velocity,
                                mol1_mass=mass[mol_i_atom_type],
                                mol2_mass=mass[mol_j_atom_type])

            # Check Hill criterion
            if pe + ke < 0:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Enforce symmetry

    return adj_matrix, pe_matrix


    
@nb.njit(parallel=True, fastmath=True)
def hill_criterion4(mol1_positions, mol2_positions, mol1_velocities, mol2_velocities,
                   atom_type_single_mol1, atom_type_single_mol2, eps, sig, q, mass,
                   box_length, cutoff):
    num_mol1 = mol1_positions.shape[0]
    num_mol2 = mol2_positions.shape[0]
    num_mol = num_mol1 + num_mol2
    
    # Total mass of each molecule
    m1 = np.sum(mass[atom_type_single_mol1])
    m2 = np.sum(mass[atom_type_single_mol2])

    mol1_com_velocities = get_com_velocity_vectorized(mol1_velocities, mass[atom_type_single_mol1])
    mol2_com_velocities = get_com_velocity_vectorized(mol2_velocities, mass[atom_type_single_mol2])

    # Initialize adjacency matrix and total potential energy
    adj_matrix = np.zeros((num_mol, num_mol), dtype=np.bool_)
    pe_matrix = np.zeros((num_mol, num_mol), dtype=np.float64)

    for i in nb.prange(num_mol):  # Parallelized loop
        if i < num_mol1:
            mol_i_atom_type = atom_type_single_mol1
            mol_i_position = mol1_positions[i]
            mol_i_com_velocity = mol1_com_velocities[i]
            mass_i = m1
        else:
            mol_i_atom_type = atom_type_single_mol2
            mol_i_position = mol2_positions[i - num_mol1]
            mol_i_com_velocity = mol2_com_velocities[i - num_mol1]
            mass_i = m2

        for j in range(i + 1, num_mol):  # Compute upper triangle only
            if j < num_mol1:
                mol_j_atom_type = atom_type_single_mol1
                mol_j_position = mol1_positions[j]
                mol_j_com_velocity = mol1_com_velocities[j]
                mass_j = m1
            else:
                mol_j_atom_type = atom_type_single_mol2
                mol_j_position = mol2_positions[j - num_mol1]
                mol_j_com_velocity = mol2_com_velocities[j - num_mol1]
                mass_j = m2

            # Compute pair potential energy
            pe = inter_molecular_pe(mol1_position=mol_i_position,
                                    mol2_position=mol_j_position,
                                    mol1_type=mol_i_atom_type,
                                    mol2_type=mol_j_atom_type,
                                    eps=eps,
                                    sig=sig,
                                    q=q,
                                    box_length=box_length,
                                    cutoff=cutoff)

            # Add to total potential energy
            # total_pe += pe
            pe_matrix[i, j] = pe
            pe_matrix[j, i] = pe

            # Only consider pairs with attractive interaction (PE < 0)
            if pe < 0:
                # Compute relative kinetic energy
                relative_velocity = mol_i_com_velocity - mol_j_com_velocity # Relative velocity
                miu = (mass_i * mass_j) / (mass_i + mass_j) # Reduced mass
                ke = 0.5 * miu * np.dot(relative_velocity, relative_velocity) * g_mol2kg * a_fs2m_s * a_fs2m_s / kcal_mol2j # Relative kinetic energy, unit: kcal_mol


                # Check Hill criterion
                if pe + ke < 0:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Enforce symmetry

    return adj_matrix, pe_matrix


