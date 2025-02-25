import numpy as np
import math
from .x_ray.AtomFormFactor import get_atomic_formfactor
from numba import jit
from numba import cuda

def get_q_vector_list_in_range(box_size_xyz_A, q_low_A, q_high_A):
    """
    Get the q vector list in the range between q_low_A and q_high_A

    :param box_size_xyz_A:
    :param q_low_A:
    :param q_high_A:
    :return:
    """

    q_min_x = np.pi * 2 / box_size_xyz_A[0]
    q_min_y = np.pi * 2 / box_size_xyz_A[1]
    q_min_z = np.pi * 2 / box_size_xyz_A[2]

    # Get the number of q to calculate
    q_num_x = int(q_high_A / q_min_x) + 1
    q_num_y = int(q_high_A / q_min_y) + 1
    q_num_z = int(q_high_A / q_min_z) + 1

    # Define a Q grid
    q_grid = np.zeros((2 * q_num_x + 1,
                       2 * q_num_y + 1,
                       2 * q_num_z + 1,
                       3), dtype=np.float64)

    q_grid[:, :, :, 0] = q_min_x * np.arange(start=-q_num_x, stop=q_num_x + 1, step=1)[:, np.newaxis, np.newaxis]
    q_grid[:, :, :, 1] = q_min_y * np.arange(start=-q_num_y, stop=q_num_y + 1, step=1)[np.newaxis, :, np.newaxis]
    q_grid[:, :, :, 2] = q_min_z * np.arange(start=-q_num_z, stop=q_num_z + 1, step=1)[np.newaxis, np.newaxis, :]

    q_length = np.linalg.norm(q_grid, axis=-1)

    # Reshape the Q grid
    q_num_tot = (2 * q_num_x + 1) * (2 * q_num_y + 1) * (2 * q_num_z + 1)
    q_grid = np.reshape(q_grid, newshape=(q_num_tot, 3))
    q_length = np.reshape(q_length, newshape=q_num_tot)

    # Get the q_list with in the range
    return np.ascontiguousarray(q_grid[(q_length < q_high_A) & (q_length > q_low_A)])


def categorize_atoms(atom_types, position_holder):
    """

    :param atom_types:
    :param position_holder:
    :return:
    """

    # Sort the atom_types array based on the type
    sorted_idx = np.argsort(atom_types)

    # Get the sorted array
    atom_type_sorted = atom_types[sorted_idx]
    position_sorted = position_holder[sorted_idx]

    # Get the index of the start and end of each kind of atoms
    atom_type_list, atom_type_start_idx, atom_type_count = np.unique(atom_type_sorted,
                                                                     return_index=True,
                                                                     return_counts=True)

    return (atom_type_list, atom_type_start_idx, atom_type_count,
            np.ascontiguousarray(atom_type_sorted), np.ascontiguousarray(position_sorted))


@cuda.jit('void(float64[:], float64[:], float64[:,:], int64[:], float64[:,:], float64[:,:], int64[:], int64, int64)')
def _get_field(cos_holder, sin_holder, unique_form_factor_list, unique_indices, q_list, atom_position, split_idx, atom_type_num, q_num):
    q_idx = cuda.grid(1)
    if q_idx < q_num:
        for atom_type_idx in range(atom_type_num):
            unique_idx = unique_indices[atom_type_idx]
            form_factor = unique_form_factor_list[unique_idx, q_idx]
            for atom_idx in range(split_idx[atom_type_idx], split_idx[atom_type_idx + 1]):
                phase = (q_list[q_idx, 0] * atom_position[atom_idx, 0] +
                         q_list[q_idx, 1] * atom_position[atom_idx, 1] +
                         q_list[q_idx, 2] * atom_position[atom_idx, 2])
                idx = atom_type_idx * q_num + q_idx
                cuda.atomic.add(cos_holder, idx, form_factor * math.cos(phase))
                cuda.atomic.add(sin_holder, idx, form_factor * math.sin(phase))

def get_field(q_list_A, atom_position_array, atom_type_array, atom_type_name_list):
    """
    Calculate the molecular dynamics form factor at a list of Q values using CUDA for each atom type.
    
    :param q_list_A: Array of Q vectors.
    :param atom_position_array: Array of atom positions.
    :param atom_type_array: Array of atom types.
    :param atom_type_name_list: List of unique atom type names.
    :return: Complex array for each atom type representing the form factor at each Q vector.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA-compatible device not available.")

    # # Get CUDA device properties
    # device = cuda.get_current_device()
    # print(f"Using GPU: {device.name}")

    # Convert the reciprocal space into a 1D series.
    q_len_array = np.linalg.norm(q_list_A, axis=-1)
    q_num = q_list_A.shape[0]
    atom_num = atom_type_array.shape[0]

    # Organize the atom info
    atom_type_unique, atom_type_start_point, atom_type_count, atom_type_sorted, atom_position_sorted = categorize_atoms(
        atom_types=atom_type_array, position_holder=atom_position_array)
    atom_type_name_list = np.array(atom_type_name_list)
    atom_type_num = len(atom_type_name_list)

    # Construct the split idx
    split_idx = np.zeros(atom_type_start_point.shape[0] + 1, dtype=np.int64)
    split_idx[:atom_type_start_point.shape[0]] = atom_type_start_point[:]
    split_idx[-1] = atom_type_array.shape[0]

    # Get unique atom type
    unique_atom_type_name_list, unique_indices = np.unique(atom_type_name_list, return_inverse=True)
    unique_atom_type_num = unique_atom_type_name_list.shape[0]

    # Get the form factor of each atom at each reciprocal point
    unique_form_factor_list = np.zeros((unique_atom_type_num, q_num), dtype=np.float64)
    for atom_type_idx in range(unique_atom_type_num):
        for q_idx in range(q_num):
            unique_form_factor_list[atom_type_idx, q_idx] = get_atomic_formfactor(
                atom_name=unique_atom_type_name_list[atom_type_idx], q_detector_in_A=q_len_array[q_idx])

    # Create holders for cos and sin components
    cos_holder = np.zeros(atom_type_num*q_num, dtype=np.float64)
    sin_holder = np.zeros(atom_type_num*q_num, dtype=np.float64)

    # Transfer data to the device
    cos_holder_device = cuda.to_device(cos_holder)
    sin_holder_device = cuda.to_device(sin_holder)
    unique_form_factor_list_device = cuda.to_device(unique_form_factor_list)
    unique_indices_device = cuda.to_device(unique_indices)
    q_list_A_device = cuda.to_device(q_list_A)
    atom_position_sorted_device = cuda.to_device(atom_position_sorted)
    split_idx_device = cuda.to_device(split_idx)

    # Calculate the pattern on the GPU
    threads_per_block = 512
    blocks_per_grid = (q_num + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    _get_field[blocks_per_grid, threads_per_block](
        cos_holder_device, sin_holder_device, unique_form_factor_list_device, unique_indices_device, q_list_A_device, 
        atom_position_sorted_device, split_idx_device, atom_type_num, q_num)

    # Ensure all CUDA operations have completed
    cuda.synchronize()

    # Copy the results back to host
    cos_holder = cos_holder_device.copy_to_host()
    sin_holder = sin_holder_device.copy_to_host()

    # Reshape 
    cos_holder = cos_holder.reshape((atom_type_num, q_num))
    sin_holder = sin_holder.reshape((atom_type_num, q_num))

    return cos_holder + 1.j * sin_holder , atom_position_sorted