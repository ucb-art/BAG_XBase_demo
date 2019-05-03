# -*- coding: utf-8 -*-

import os
import importlib
from itertools import product

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

from bag import BagProject
from bag.io import read_yaml
from bag.layout.routing import RoutingGrid
from bag.layout.template import TemplateDB
from bag.data import load_sim_results, save_sim_results, load_sim_file

from xbase_demo.demo_layout.core import RoutingDemo


def make_tdb(prj, specs, impl_lib):
    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']

    # create RoutingGrid object
    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    # create layout template database
    tdb = TemplateDB('template_libs.def', routing_grid, impl_lib, use_cybagoa=True)
    return tdb


def gen_pwl_data(fname):
    td = 100e-12
    tpulse = 800e-12
    tr = 20e-12
    amp = 10e-3

    tvec = [0, td, td + tr, td + tr + tpulse, td + tr + tpulse + tr]
    yvec = [-amp, -amp, amp, amp, -amp]

    dir_name = os.path.dirname(fname)
    os.makedirs(dir_name, exist_ok=True)

    with open(fname, 'w') as f:
        for t, y in zip(tvec, yvec):
            f.write('%.4g %.4g\n' % (t, y))


def routing_demo(prj, specs):
    impl_lib = 'DEMO_ROUTING'

    # create layout template database
    tdb = make_tdb(prj, specs, impl_lib)
    # compute layout
    print('computing layout')
    template = tdb.new_template(params={}, temp_cls=RoutingDemo)
    # create layout in OA database
    print('creating layout')
    tdb.batch_layout(prj, [template], ['ROUTING_DEMO'])
    # return corresponding schematic parameters
    print('layout done')


def gen_layout(prj, specs, dsn_name):
    # get information from specs
    dsn_specs = specs[dsn_name]
    impl_lib = dsn_specs['impl_lib']
    layout_params = dsn_specs['layout_params']
    lay_package = dsn_specs['layout_package']
    lay_class = dsn_specs['layout_class']
    gen_cell = dsn_specs['gen_cell']

    # get layout generator class
    lay_module = importlib.import_module(lay_package)
    temp_cls = getattr(lay_module, lay_class)

    # create layout template database
    tdb = make_tdb(prj, specs, impl_lib)
    # compute layout
    print('computing layout')
    template = tdb.new_template(params=layout_params, temp_cls=temp_cls)
    # create layout in OA database
    print('creating layout')
    tdb.batch_layout(prj, [template], [gen_cell])
    # return corresponding schematic parameters
    print('layout done')
    return template.sch_params


def gen_schematics(prj, specs, dsn_name, sch_params, check_lvs=False):
    dsn_specs = specs[dsn_name]

    impl_lib = dsn_specs['impl_lib']
    sch_lib = dsn_specs['sch_lib']
    sch_cell = dsn_specs['sch_cell']
    gen_cell = dsn_specs['gen_cell']
    testbenches = dsn_specs['testbenches']

    # create schematic generator object
    dsn = prj.create_design_module(sch_lib, sch_cell)
    # compute schematic
    print('computing %s schematics' % gen_cell)
    dsn.design(**sch_params)
    # create schematic in OA database
    print('creating %s schematics' % gen_cell)
    dsn.implement_design(impl_lib, top_cell_name=gen_cell, erase=True)

    if check_lvs:
        print('running lvs')
        lvs_passed, lvs_log = prj.run_lvs(impl_lib, gen_cell)
        if not lvs_passed:
            raise ValueError('LVS failed.  check log file: %s' % lvs_log)
        else:
            print('lvs passed')
            print('lvs log is ' + lvs_log)

    for name, info in testbenches.items():
        tb_lib = info['tb_lib']
        tb_cell = info['tb_cell']
        tb_sch_params = info['sch_params']

        tb_gen_cell = '%s_%s' % (gen_cell, name)

        if 'tran_fname' in tb_sch_params:
            tran_fname = os.path.abspath(tb_sch_params['tran_fname'])
            gen_pwl_data(tran_fname)
            tb_sch_params['tran_fname'] = tran_fname

        tb_dsn = prj.create_design_module(tb_lib, tb_cell)
        print('computing %s schematics' % tb_gen_cell)
        tb_dsn.design(dut_lib=impl_lib, dut_cell=gen_cell, **tb_sch_params)
        print('creating %s schematics' % tb_gen_cell)
        tb_dsn.implement_design(impl_lib, top_cell_name=tb_gen_cell, erase=True)

    print('schematic done')


def simulate(prj, specs, dsn_name):
    view_name = specs['view_name']
    sim_envs = specs['sim_envs']
    dsn_specs = specs[dsn_name]

    data_dir = dsn_specs['data_dir']
    impl_lib = dsn_specs['impl_lib']
    gen_cell = dsn_specs['gen_cell']
    testbenches = dsn_specs['testbenches']

    results_dict = {}
    for name, info in testbenches.items():
        tb_params = info['tb_params']
        tb_gen_cell = '%s_%s' % (gen_cell, name)

        # setup testbench ADEXL state
        print('setting up %s' % tb_gen_cell)
        tb = prj.configure_testbench(impl_lib, tb_gen_cell)
        # set testbench parameters values
        for key, val in tb_params.items():
            tb.set_parameter(key, val)
        # set config view, i.e. schematic vs extracted
        tb.set_simulation_view(impl_lib, gen_cell, view_name)
        # set process corners
        tb.set_simulation_environments(sim_envs)
        # commit changes to ADEXL state back to database
        tb.update_testbench()
        # start simulation
        print('running simulation')
        tb.run_simulation()
        # import simulation results to Python
        print('simulation done, load results')
        results = load_sim_results(tb.save_dir)
        # save simulation data as HDF5 format
        save_sim_results(results, os.path.join(data_dir, '%s.hdf5' % tb_gen_cell))

        results_dict[name] = results

    print('all simulation done')

    return results_dict


def load_sim_data(specs, dsn_name):
    dsn_specs = specs[dsn_name]
    data_dir = dsn_specs['data_dir']
    gen_cell = dsn_specs['gen_cell']
    testbenches = dsn_specs['testbenches']

    results_dict = {}
    for name, info in testbenches.items():
        tb_gen_cell = '%s_%s' % (gen_cell, name)
        fname = os.path.join(data_dir, '%s.hdf5' % tb_gen_cell)
        print('loading simulation data for %s' % tb_gen_cell)
        results_dict[name] = load_sim_file(fname)

    print('finish loading data')

    return results_dict


def split_data_by_sweep(results, var_list):
    sweep_names = results['sweep_params'][var_list[0]][:-1]
    combo_list = []
    for name in sweep_names:
        combo_list.append(range(results[name].size))

    if combo_list:
        idx_list_iter = product(*combo_list)
    else:
        idx_list_iter = [[]]

    ans_list = []
    for idx_list in idx_list_iter:
        cur_label_list = []
        for name, idx in zip(sweep_names, idx_list):
            swp_val = results[name][idx]
            if isinstance(swp_val, str):
                cur_label_list.append('%s=%s' % (name, swp_val))
            else:
                cur_label_list.append('%s=%.4g' % (name, swp_val))

        if cur_label_list:
            label = ', '.join(cur_label_list)
        else:
            label = ''

        cur_idx_list = list(idx_list)
        cur_idx_list.append(slice(None))

        cur_results = {var: results[var][cur_idx_list] for var in var_list}
        ans_list.append((label, cur_results))

    return ans_list


def process_tb_dc(tb_results, plot=True):
    result_list = split_data_by_sweep(tb_results, ['vin', 'vout'])

    plot_data_list = []
    for label, res_dict in result_list:
        cur_vin = res_dict['vin']
        cur_vout = res_dict['vout']

        cur_vin, vin_arg = np.unique(cur_vin, return_index=True)
        cur_vout = cur_vout[vin_arg]
        vout_fun = interp.InterpolatedUnivariateSpline(cur_vin, cur_vout)
        vout_diff_fun = vout_fun.derivative(1)

        print('%s, gain=%.4g' % (label, vout_diff_fun([0])))
        plot_data_list.append((label, cur_vin, cur_vout, vout_diff_fun(cur_vin)))

    if plot:
        f, (ax1, ax2) = plt.subplots(2, sharex='all')
        ax1.set_title('Vout vs Vin')
        ax1.set_ylabel('Vout (V)')
        ax2.set_title('Gain vs Vin')
        ax2.set_ylabel('Gain (V/V)')
        ax2.set_xlabel('Vin (V)')

        for label, vin, vout, vdiff in plot_data_list:
            if label:
                ax1.plot(cur_vin, cur_vout, label=label)
                ax2.plot(cur_vin, vout_diff_fun(cur_vin), label=label)
            else:
                ax1.plot(cur_vin, cur_vout)
                ax2.plot(cur_vin, vout_diff_fun(cur_vin))

        if len(result_list) > 1:
            ax1.legend()
            ax2.legend()


def process_tb_ac(tb_results, plot=True):
    result_list = split_data_by_sweep(tb_results, ['vout_ac'])

    freq = tb_results['freq']
    log_freq = np.log10(freq)
    plot_data_list = []
    for label, res_dict in result_list:
        cur_vout = res_dict['vout_ac']
        cur_mag = 20 * np.log10(np.abs(cur_vout))  # type: np.ndarray
        cur_ang = np.angle(cur_vout, deg=True)

        # interpolate log-log plot
        mag_fun = interp.InterpolatedUnivariateSpline(log_freq, cur_mag)
        ang_fun = interp.InterpolatedUnivariateSpline(log_freq, cur_ang)
        # find 3db and unity gain frequency
        dc_gain = cur_mag[0]
        lf0 = log_freq[0]
        lf1 = log_freq[-1]
        try:
            lf_3db = sciopt.brentq(lambda x: mag_fun(x) - (dc_gain - 3), lf0, lf1)  # type: float
            freq_3db = 10.0**lf_3db
        except ValueError:
            lf_3db = 0
            freq_3db = -1
        try:
            # noinspection PyTypeChecker
            lf_unity = sciopt.brentq(mag_fun, lf0, lf1)  # type: float
            freq_unity = 10.0**lf_unity
        except ValueError:
            lf_unity = 0
            freq_unity = -1

        # find phase margin
        if freq_unity > 0:
            pm = 180 + ang_fun(lf_unity) - ang_fun(lf0)
        else:
            pm = 360

        print('%s, f_3db=%.4g, f_unity=%.4g, phase_margin=%.4g' % (label, freq_3db, freq_unity, pm))
        plot_data_list.append((label, cur_mag, cur_ang))

    if plot:
        f, (ax1, ax2) = plt.subplots(2, sharex='all')
        ax1.set_title('Magnitude vs Frequency')
        ax1.set_ylabel('Magnitude (dB)')
        ax2.set_title('Phase vs Frequency')
        ax2.set_ylabel('Phase (Degrees)')
        ax2.set_xlabel('Frequency (Hz)')

        for label, cur_mag, cur_ang in plot_data_list:
            if label:
                ax1.semilogx(freq, cur_mag, label=label)
                ax2.semilogx(freq, cur_ang, label=label)
            else:
                ax1.semilogx(freq, cur_mag)
                ax2.semilogx(freq, cur_ang)

        if len(result_list) > 1:
            ax1.legend()
            ax2.legend()


def process_tb_tran(tb_results, plot=True):
    result_list = split_data_by_sweep(tb_results, ['vout_tran'])

    tvec = tb_results['time']
    plot_data_list = []
    for label, res_dict in result_list:
        cur_vout = res_dict['vout_tran']

        plot_data_list.append((label, cur_vout))

    if plot:
        plt.figure()
        plt.title('Vout vs Time')
        plt.ylabel('Vout (V)')
        plt.xlabel('Time (s)')

        for label, cur_vout in plot_data_list:
            if label:
                plt.plot(tvec, cur_vout, label=label)
            else:
                plt.plot(tvec, cur_vout)

        if len(result_list) > 1:
            plt.legend()


def plot_data(results_dict, plot=True):
    process_tb_dc(results_dict['tb_dc'], plot=plot)
    process_tb_ac(results_dict['tb_ac_tran'], plot=plot)
    process_tb_tran(results_dict['tb_ac_tran'], plot=plot)

    if plot:
        plt.show()


def run_flow(prj, specs, dsn_name):
    run_lvs = True

    # generate layout, get schematic parameters from layout
    dsn_sch_params = gen_layout(prj, specs, dsn_name)
    # generate design/testbench schematics
    gen_schematics(prj, specs, dsn_name, dsn_sch_params, check_lvs=run_lvs)
    # run simulation and import results
    simulate(prj, specs, dsn_name)

    # load simulation results from save file
    res_dict = load_sim_data(specs, dsn_name)
    # post-process simulation results
    plot_data(res_dict)


if __name__ == '__main__':
    spec_fname = 'specs_demo/demo.yaml'

    # load specifications from file
    top_specs = read_yaml(spec_fname)

    # create BagProject object
    local_dict = locals()
    if 'bprj' in local_dict:
        print('using existing BagProject')
        bprj = local_dict['bprj']
    else:
        print('creating BagProject')
        bprj = BagProject()

    # routing_demo(bprj, top_specs)
    run_flow(bprj, top_specs, 'amp_cs')
    # gen_layout(bprj, top_specs, 'amp_sf')
    # bprj.import_design_library('demo_templates')
    # run_flow(bprj, top_specs, 'amp_sf')
    # gen_layout(bprj, top_specs, 'amp_chain')
    # run_flow(bprj, top_specs, 'amp_chain')
