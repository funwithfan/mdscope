"""This module contains functions for reading and writing LAMMPS dump files."""

import numpy as np
import pandas as pd

def extract_lammps_data(filename, columns=None, filters=None, sort_by=None):
    """
    Extract specific information from a LAMMPS dump file, including box size limits.
    
    Parameters:
        filename (str): Path to the LAMMPS dump file.
        columns (list, optional): List of column names to extract. Defaults to all columns.
        filters (dict, optional): Dictionary specifying filters in the format {column: value}.
            Can use tuples (min, max) for range filtering.
        sort_by (str, optional): Column name to sort the data by.
    
    Returns:
        tuple: (pd.DataFrame with extracted data, np.array with box size limits (3x2))
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract box size limits
    box_limits = np.zeros((3, 2))
    for i, line in enumerate(lines):
        if line.startswith("ITEM: BOX BOUNDS"):
            box_limits[0] = list(map(float, lines[i + 1].split()))
            box_limits[1] = list(map(float, lines[i + 2].split()))
            box_limits[2] = list(map(float, lines[i + 3].split()))
        
        if line.startswith("ITEM: ATOMS") or line.startswith("ITEM: ENTRIES"):
            start_index = i + 1
            column_names = line.strip().split()[2:]
            break
    else:
        raise ValueError("ATOM data section not found in the dump file.")
    
    data_lines = lines[start_index:]
    
    # Read data into a DataFrame
    data = pd.read_csv(pd.io.common.StringIO(''.join(data_lines)), sep=r"\s+", names=column_names)
    
    # Use all columns if not specified
    if columns is None:
        columns = column_names
    
    # Filter columns
    data = data[columns]
    
    # Apply filters if provided
    if filters:
        for key, value in filters.items():
            if isinstance(value, tuple):  # Range filter
                data = data[(data[key] >= value[0]) & (data[key] <= value[1])]
            else:  # Exact match filter
                data = data[data[key] == value]
    
    # Sort data
    if sort_by:
        if sort_by in data.columns:
            data = data.sort_values(by=sort_by)
        else:
            print('Warning: sort_by is not in data. Returning unsorted data.')
            
    return data, box_limits


def write_lammps_dump(filename, timestep, num_atoms, box_limits, df, order):
    """
    Write a LAMMPS dump file.

    Parameters:
        filename (str): Output file name.
        timestep (int): Timestep number.
        num_atoms (int): Number of atoms.
        box_limits (numpy.ndarray): 3x2 array of box boundaries.
        df (pandas.DataFrame): DataFrame containing atomic data.
        order (list): Column names in the order they should be written.
    """

    # Ensure `order` is a list
    if not isinstance(order, list):
        order = list(order)

    # Validate `box_limits` shape
    if not (isinstance(box_limits, np.ndarray) and box_limits.shape == (3, 2)):
        raise ValueError("box_limits must be a (3,2) NumPy array.")

    # Check if all columns in `order` exist in `df`, if not raise an error
    missing_columns = [col for col in order if col not in df.columns]
    if missing_columns:
        raise KeyError(f"DataFrame is missing required columns: {missing_columns}")

    with open(filename, 'w') as file:
        # Write timestep
        file.write("ITEM: TIMESTEP\n")
        file.write(f"{timestep}\n")

        # Write number of atoms
        file.write("ITEM: NUMBER OF ATOMS\n")
        file.write(f"{num_atoms}\n")

        # Write box limits
        file.write("ITEM: BOX BOUNDS pp pp pp\n")
        for i in range(3):
            file.write(f"{box_limits[i, 0]} {box_limits[i, 1]}\n")
        
        # Write column names in the specified order
        file.write("ITEM: ATOMS ")
        file.write(' '.join(order) + '\n')

        # Write atomic data with specified column order
        df.to_csv(file, sep=' ', header=False, index=False, columns=order)