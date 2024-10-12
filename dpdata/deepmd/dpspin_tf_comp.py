from __future__ import annotations

import glob
import os
import shutil
import warnings

import numpy as np

import dpdata
from dpdata.utils import open_file



def _cond_load_data(fname):
    tmp = None
    if os.path.isfile(fname):
        tmp = np.load(fname)
    return tmp


def _load_set(folder, nopbc: bool):
    coords_spins = np.load(os.path.join(folder, "coord.npy"))
    if nopbc:
        cells = np.zeros((coords.shape[0], 3, 3))
    else:
        cells = np.load(os.path.join(folder, "box.npy"))
    return cells, coords_spins


def load_type(folder, spin_norm, virtual_len, type_map=None):
    data = {}
    atom_types = np.loadtxt(os.path.join(folder, "type.raw"), ndmin=1).astype(int)
    ntypes_spin = len(spin_norm)
    ntypes = np.max(atom_types) + 1 - ntypes_spin
    assert len(spin_norm) == len(virtual_len), f"The length of spin_norm should be equal to virtual length."
    assert len(spin_norm) <= ntypes, f"The length of spin_norm should be no more than ntypes {ntypes}."

    data["atom_types"] = atom_types[~np.isin(atom_types, np.arange(ntypes, ntypes + ntypes_spin))]
    
    data["atom_names"] = []
    # if find type_map.raw, use it
    if os.path.isfile(os.path.join(folder, "type_map.raw")):
        with open_file(os.path.join(folder, "type_map.raw")) as fp:
            my_type_map = fp.read().split()
    # else try to use arg type_map
    elif type_map is not None:
        my_type_map = type_map
    # in the last case, make artificial atom names
    else:
        my_type_map = []
        for ii in range(ntypes):
            my_type_map.append("Type_%d" % ii)
    data["atom_names"] = my_type_map
    data["atom_numbs"] = []
    for ii, _ in enumerate(data["atom_names"]):
        data["atom_numbs"].append(np.count_nonzero(data["atom_types"] == ii))

    return data


def to_system_data(folder, spin_norm, virtual_len, type_map=None, labels=True):
    # data is empty
    data = load_type(folder, spin_norm, virtual_len, type_map=type_map)
    data["orig"] = np.zeros([3])
    if os.path.isfile(os.path.join(folder, "nopbc")):
        data["nopbc"] = True
    sets = sorted(glob.glob(os.path.join(folder, "set.*")))
    all_cells = []
    all_coords = []
    all_spins = []
    all_forces = []
    all_mag_forces = []
    ntypes = np.max(data["atom_types"]) + 1
    natoms = data["atom_types"].shape[0]
    spin_norm = np.concatenate((spin_norm, np.full(ntypes - spin_norm.shape, -1)))
    virtual_len = np.concatenate((virtual_len, np.full(ntypes - virtual_len.shape, -1)))
    for ii in sets:
        cells, coords_spins = _load_set(ii, data.get("nopbc", False))
        nframes = np.reshape(cells, [-1, 3, 3]).shape[0]
        coords_spins = coords_spins.reshape([nframes, -1, 3])
        forces_total = _cond_load_data(os.path.join(ii, "force.npy")).reshape((nframes, -1, 3))

        coords = coords_spins[:, :natoms, :].copy()
        forces = forces_total[:, :natoms, :].copy()
        spins = np.zeros((nframes, natoms, 3))
        mag_forces = np.zeros((nframes, natoms, 3))
        current_spin_index = natoms
        for atom_type in range(ntypes):
            mask = (data["atom_types"] == atom_type)
            if spin_norm[atom_type] != -1 and virtual_len[atom_type] != -1:
                # Calculates the spins
                n_spins_to_add = data["atom_numbs"][atom_type]
                spins[:, mask] = (coords_spins[:, current_spin_index:current_spin_index + n_spins_to_add] - coords[:, mask]) *  spin_norm[atom_type] / virtual_len[atom_type]
                mag_forces[:, mask] = forces_total[:, current_spin_index:current_spin_index + n_spins_to_add]           
                current_spin_index += n_spins_to_add

        all_cells.append(np.reshape(cells, [nframes, 3, 3]))
        all_coords.append(coords)
        all_forces.append(forces)
        all_spins.append(spins)
        all_mag_forces.append(mag_forces)

    data["cells"] = np.concatenate(all_cells, axis=0)
    data["coords"] = np.concatenate(all_coords, axis=0)
    data["forces"] = np.concatenate(all_forces, axis=0)
    data["spins"] = np.concatenate(all_spins, axis=0)
    data["mag_forces"] = np.concatenate(all_mag_forces, axis=0)
    # allow custom dtypes
    if labels:
        dtypes = dpdata.system.LabeledSystem.DTYPES
    else:
        dtypes = dpdata.system.System.DTYPES

    for dtype in dtypes:
        if dtype.name in (
            "atom_numbs",
            "atom_names",
            "atom_types",
            "orig",
            "cells",
            "coords",
            "spins",
            "forces",
            "mag_forces",
            "real_atom_names",
            "nopbc",
        ):
            # skip as these data contains specific rules
            continue
        if not (len(dtype.shape) and dtype.shape[0] == dpdata.system.Axis.NFRAMES):
            warnings.warn(
                f"Shape of {dtype.name} is not (nframes, ...), but {dtype.shape}. This type of data will not converted from deepmd/npy format."
            )
            continue
        shape = [
            natoms if xx == dpdata.system.Axis.NATOMS else xx for xx in dtype.shape[1:]
        ]
        all_data = []
        for ii in sets:
            tmp = _cond_load_data(os.path.join(ii, dtype.deepmd_name + ".npy"))
            if tmp is not None:
                all_data.append(np.reshape(tmp, [tmp.shape[0], *shape]))
        if len(all_data) > 0:
            data[dtype.name] = np.concatenate(all_data, axis=0)
    return data


def dump(folder, data, spin_norm, virtual_len, set_size=5000, comp_prec=np.float32, remove_sets=True):
    os.makedirs(folder, exist_ok=True)
    sets = sorted(glob.glob(os.path.join(folder, "set.*")))
    if len(sets) > 0:
        if remove_sets:
            for ii in sets:
                shutil.rmtree(ii)
        else:
            raise RuntimeError(
                "found "
                + str(sets)
                + " in "
                + folder
                + "not a clean deepmd raw dir. please firstly clean set.* then try compress"
            )
    # dump raw
    # np.savetxt(os.path.join(folder, "type.raw"), data["atom_types"], fmt="%d")
    # np.savetxt(os.path.join(folder, "type_map.raw"), data["atom_names"], fmt="%s")
    # BondOrder System
    if "bonds" in data:
        np.savetxt(
            os.path.join(folder, "bonds.raw"),
            data["bonds"],
            header="begin_atom, end_atom, bond_order",
        )
    if "formal_charges" in data:
        np.savetxt(os.path.join(folder, "formal_charges.raw"), data["formal_charges"])
    # reshape frame properties and convert prec
    nframes = data["cells"].shape[0]
    # dump frame properties: cell, coord, energy, force and virial
    nsets = nframes // set_size
    if set_size * nsets < nframes:
        nsets += 1
    for ii in range(nsets):
        set_stt = ii * set_size
        set_end = (ii + 1) * set_size
        set_folder = os.path.join(folder, "set.%03d" % ii)
        os.makedirs(set_folder)
    try:
        os.remove(os.path.join(folder, "nopbc"))
    except OSError:
        pass
    if data.get("nopbc", False):
        with open_file(os.path.join(folder, "nopbc"), "w") as fw_nopbc:
            pass
    # allow custom dtypes
    labels = "energies" in data
    if labels:
        dtypes = dpdata.system.LabeledSystem.DTYPES
    else:
        dtypes = dpdata.system.System.DTYPES
    for dtype in dtypes:
        if dtype.name in (
            "atom_numbs",
            "atom_names",
            "atom_types",
            "orig",
            "real_atom_names",
            "nopbc",
            "coords",
            "forces",
            "spins",
            "mag_forces",
        ):
            # skip as these data contains specific rules
            continue
        if dtype.name not in data:
            continue
        if not (len(dtype.shape) and dtype.shape[0] == dpdata.system.Axis.NFRAMES):
            warnings.warn(
                f"Shape of {dtype.name} is not (nframes, ...), but {dtype.shape}. This type of data will not converted to deepmd/npy format."
            )
            continue
        ddata = np.reshape(data[dtype.name], [nframes, -1])
        if np.issubdtype(ddata.dtype, np.floating):
            ddata = ddata.astype(comp_prec)
        for ii in range(nsets):
            set_stt = ii * set_size
            set_end = (ii + 1) * set_size
            set_folder = os.path.join(folder, "set.%03d" % ii)
            np.save(os.path.join(set_folder, dtype.deepmd_name), ddata[set_stt:set_end])

    natoms = data["coords"].shape[1]
    ntypes = np.max(data["atom_types"]) + 1
    assert len(spin_norm) == len(virtual_len), f"The length of spin_norm should be equal to virtual length."
    assert len(spin_norm) <= ntypes, f"The length of spin_norm should be no more than ntypes {ntypes}."
    spin_norm = np.concatenate((spin_norm, np.full(ntypes - spin_norm.shape, -1)))
    virtual_len = np.concatenate((virtual_len, np.full(ntypes - virtual_len.shape, -1)))

    # Calculate the number of spins
    valid_spin_count = np.sum(data["atom_numbs"] * ((spin_norm != -1) & (virtual_len != -1)))

    # Initializes the result array
    coords = np.zeros((nframes, natoms + valid_spin_count, 3))
    forces = np.zeros((nframes, natoms + valid_spin_count, 3))
    atom_types = np.zeros((natoms + valid_spin_count))
    coords[:, :natoms] = data["coords"].copy()
    forces[:, :natoms] = data["forces"].copy()
    atom_types[:natoms] = data["atom_types"].copy() 

    # Record the current spin insertion position and atom type offset
    current_spin_index = natoms
    new_atom_type_offset = ntypes

    for atom_type in range(ntypes):
        mask = (data["atom_types"] == atom_type)
        if spin_norm[atom_type] != -1 and virtual_len[atom_type] != -1:
            # Calculates the spins, and updates the results and atom_types
            spins_to_add = data["spins"][:, mask] * virtual_len[atom_type] / spin_norm[atom_type] + data["coords"][:, mask]
            n_spins_to_add = data["atom_numbs"][atom_type]

            coords[:, current_spin_index:current_spin_index + n_spins_to_add] = spins_to_add
            forces[:, current_spin_index:current_spin_index + n_spins_to_add] = data["mag_forces"][:, mask].copy()
            atom_types[current_spin_index:current_spin_index + n_spins_to_add] = np.full(n_spins_to_add, new_atom_type_offset)

            current_spin_index += n_spins_to_add
            new_atom_type_offset += 1

    np.savetxt(os.path.join(folder, "type.raw"), atom_types, fmt="%d")

    coords = coords.reshape([nframes, -1]).astype(comp_prec)
    forces = forces.reshape([nframes, -1]).astype(comp_prec)
    for ii in range(nsets):
        set_stt = ii * set_size
        set_end = (ii + 1) * set_size
        set_folder = os.path.join(folder, "set.%03d" % ii)
        np.save(os.path.join(set_folder, "coord.npy"), coords[set_stt:set_end])
        np.save(os.path.join(set_folder, "force.npy"), forces[set_stt:set_end])
