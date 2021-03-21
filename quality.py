from .xharpy import calc_f, construct_values
import numpy as np

def calculate_quality_indicators(construction_instructions, parameters, fjs, cell_mat_m, symm_mats_vecs, index_vec_h, intensities, stderr):
    cell_mat_f = np.linalg.inv(cell_mat_m)
    xyz, uij, cijk, dijkl, occupancies = construct_values(parameters, construction_instructions, cell_mat_m)

    structure_factors = calc_f(
        xyz=xyz,
        uij=uij,
        cijk=cijk,
        dijkl=dijkl,
        occupancies=occupancies,
        index_vec_h=index_vec_h,
        cell_mat_f=cell_mat_f,
        symm_mats_vecs=symm_mats_vecs,
        fjs=fjs
    )

    f_obs = np.sign(intensities) * np.sqrt(np.abs(intensities))
    f_obs_safe = np.array(f_obs)
    f_obs_safe[f_obs_safe == 0] = 1e-20
    i_over_2sigma = intensities / stderr > 2

    r_f = np.sum(np.abs(f_obs - np.sqrt(parameters[0]) * np.abs(structure_factors))) / np.sum(np.abs(f_obs))
    r_f_strong = np.sum(np.abs(f_obs[i_over_2sigma] - np.sqrt(parameters[0]) * np.abs(structure_factors[i_over_2sigma]))) / np.sum(np.abs(f_obs[i_over_2sigma]))
    r_f2 = np.sum(np.abs(intensities - parameters[0] * np.abs(structure_factors)**2)) / np.sum(intensities)
    wr2 = np.sqrt(np.sum(1/stderr**2 * (intensities - parameters[0] *  np.abs(structure_factors)**2)**2) / np.sum(1/stderr**2 * intensities**2))
    gof = np.sum(1/stderr**2 * (intensities - parameters[0] * np.abs(structure_factors)**2)**2) / (len(intensities) - len(parameters))

    return {
            'R(F)': r_f,
            'R(F)(I>2s)': r_f_strong,
            'R(F^2)': r_f2,
            'wR(F^2)': wr2,
            'GOF': gof
    }
