from .xharpy import calc_f, construct_values
from .conversion import calc_sin_theta_ov_lambda, cell_constants_to_M, calc_sin_theta_ov_lambda
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


def calculate_drk(df, bins=21, equal_sized_bins=False, cell_mat_m=None):
    df = df.copy()
    df = df.rename({
        'refln_index_h': 'h',
        'refln_index_k': 'k',
        'refln_index_l': 'l',
        'refln_F_squared_meas': 'intensity',
        'refln_F_squared_sigma': 'stderr',
        'refln_F_squared_calc': 'intensity_calc',
        'refln_F_calc': 'f_calc',
        'refln_phase_calc': 'phase_calc',
        'refln_sint/lambda': 'sint/lambda'
    }, axis=1)
    if 'intensity_calc' not in df.columns:
        df['intensity_calc'] = np.abs(df['f_calc'])**2
    if 'sint/lambda' not in df.columns:
        assert cell_mat_m is not None, 'You either need to give cell_mat_m or sint/lambda in the df'
        cell_mat_f = np.linalg.inv(cell_mat_m)
        df['sint/lambda'] = np.array(calc_sin_theta_ov_lambda(cell_mat_f, df[['h', 'k', 'l']].values))
    if equal_sized_bins:
        try:
            n_bins = int(bins)
        except (TypeError, ValueError):
            assert not isinstance(bins, str)
            raise TypeError('bins has to be an integer number ')
        sort = df.sort_values('sint/lambda')
        splits = np.array_split(df.sort_values('sint/lambda'), bins)
        mean_res = [np.mean(split['sint/lambda']) for split in splits]
        drk = [np.sum(split['intensity']) / np.sum(split['intensity_calc']) for split in splits]
        count = [len(split) for split in splits]
    else:
        # non equal bins
        try:
            n_bins = int(bins)
            assert float(bins) == int(bins) # check if float
            # number of bins given as integer
            res_min = df['sint/lambda'].min()
            res_max = df['sint/lambda'].max()
            limits = np.linspace(res_min, res_max, n_bins + 1)
        except (TypeError, ValueError):
            try:
                assert not isinstance(bins, str) # this would create a list of chars
                limits = list(bins)
            except (TypeError, AssertionError):
                raise TypeError('bins has to be either an integer number or a list-like')
        except AssertionError:
            # bins is float so a step size
            step = float(bins)
            lower = np.floor(df['sint/lambda'].min() / step) * step
            limits = np.arange(lower, df['sint/lambda'].max() + step, step)
            
        conditions = [np.logical_and(df['sint/lambda'] > lim_low, df['sint/lambda'] < lim_high) for lim_low, lim_high in zip(limits[:-1], limits[1:])]
        drk = [np.sum(df[condition]['intensity']) / np.sum(df[condition]['intensity_calc']) for condition in conditions]
        count = [len(df[condition]) for condition in conditions]
    result = {
        'Sum(F^2_Obs)/Sum(F^2_calc)': np.array(drk),
        'count': np.array(count)
    }
    if equal_sized_bins:
        result['mean resolution'] = np.array(mean_res)
    else:
        result['lower limit'] = np.array(limits[:-1])
        result['upper limit'] = np.array(limits[1:])
    return result