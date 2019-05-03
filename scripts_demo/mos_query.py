# -*- coding: utf-8 -*-

import pprint

import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import ticker

from ckt_dsn.mos.core import MOSDBDiscrete


def query(vds, vgs, env_list=None, intent='lvt', vbs=0.0):
    """Get interpolation function and plot/query."""
    w_list = [4]
    w_res = 1
    interp_method = 'spline'
    cfit_method = 'average'
    spec_list = ['mos_data/mos_char_nch/specs.yaml']

    if env_list is None:
        env_list = ['tt']

    # initialize transistor database from simulation data
    nch_db = MOSDBDiscrete(w_list, spec_list, w_res,
                           method=interp_method,
                           cfit_method=cfit_method)
    # set process corners
    nch_db.env_list = env_list
    # set layout parameters
    nch_db.set_dsn_params(w=w_list[0], intent=intent)
    # returns a dictionary of smal-signal parameters
    return nch_db.query(vbs=vbs, vds=vds, vgs=vgs)


def plot_data(name='ibias', vbs=0.0, intent='lvt', env_list=None,
              bounds=None, unit_val=None, unit_label=None):
    """Get interpolation function and plot/query."""
    w_list = [4]
    w_res = 1
    interp_method = 'spline'
    cfit_method = 'average'
    spec_list = ['mos_data/mos_char_nch/specs.yaml']

    if env_list is None:
        env_list = ['tt', 'ff_hot', 'ss_cold']

    nvds = 41
    nvgs = 81

    print('create transistor database')
    nch_db = MOSDBDiscrete(w_list, spec_list, w_res,
                           method=interp_method,
                           cfit_method=cfit_method)
    nch_db.env_list = env_list
    nch_db.set_dsn_params(w=w_list[0], intent=intent)

    f = nch_db.get_function(name)
    vds_min, vds_max = f.get_input_range(1)
    vgs_min, vgs_max = f.get_input_range(2)
    if bounds is not None:
        if 'vgs' in bounds:
            v0, v1 = bounds['vgs']
            if v0 is not None:
                vgs_min = max(vgs_min, v0)
            if v1 is not None:
                vgs_max = min(vgs_max, v1)
        if 'vds' in bounds:
            v0, v1 = bounds['vds']
            if v0 is not None:
                vds_min = max(vds_min, v0)
            if v1 is not None:
                vds_max = min(vds_max, v1)

    vbs_vec = [vbs]
    vds_vec = np.linspace(vds_min, vds_max, nvds, endpoint=True)
    vgs_vec = np.linspace(vgs_min, vgs_max, nvgs, endpoint=True)
    vbs_mat, vds_mat, vgs_mat = np.meshgrid(vbs_vec, vds_vec, vgs_vec, indexing='ij', copy=False)
    arg = np.stack((vbs_mat, vds_mat, vgs_mat), axis=-1)
    ans = f(arg)

    vds_mat = vds_mat.reshape((nvds, nvgs))
    vgs_mat = vgs_mat.reshape((nvds, nvgs))
    ans = ans.reshape((nvds, nvgs, len(env_list)))

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    if unit_label is not None:
        zlabel = '%s (%s)' % (name, unit_label)
    else:
        zlabel = name
    for idx, env in enumerate(env_list):
        fig = plt.figure(idx + 1)
        ax = fig.add_subplot(111, projection='3d')
        cur_val = ans[..., idx]
        if unit_val is not None:
            cur_val = cur_val / unit_val
        ax.plot_surface(vds_mat, vgs_mat, cur_val, rstride=1, cstride=1, linewidth=0, cmap=cm.cubehelix)
        ax.set_title('%s (corner=%s)' % (name, env))
        ax.set_xlabel('Vds (V)')
        ax.set_ylabel('Vgs (V)')
        ax.set_zlabel(zlabel)
        ax.w_zaxis.set_major_formatter(formatter)

    plt.show()
