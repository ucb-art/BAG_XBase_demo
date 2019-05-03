# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from bag.layout.routing import TrackID
from bag.layout.template import TemplateBase

from abs_templates_ec.analog_core import AnalogBase


class RoutingDemo(TemplateBase):
    """A template of a single transistor with dummies.

    This class is mainly used for transistor characterization or
    design exploration with config views.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        super(RoutingDemo, self).__init__(temp_db, lib_name, params, used_names, **kwargs)

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return {}

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        # Metal 4 is horizontal, Metal 5 is vertical
        hm_layer = 4
        vm_layer = 5

        # add a horizontal wire on track 0, from X=0.1 to X=0.3
        warr1 = self.add_wires(hm_layer, 0, 0.1, 0.3)
        # print WireArray object
        print(warr1)
        # print lower, middle, and upper coordinate of wire.
        print(warr1.lower, warr1.middle, warr1.upper)
        # print TrackID object associated with WireArray
        print(warr1.track_id)
        # add a horizontal wire on track 1, from X=0.1 to X=0.3,
        # coordinates specified in resolution units
        warr2 = self.add_wires(hm_layer, 1, 100, 300, unit_mode=True)
        # add a horizontal wire on track 2.5, from X=0.2 to X=0.4
        self.add_wires(hm_layer, 2.5, 200, 400, unit_mode=True)
        # add a horizontal wire on track 4, from X=0.2 to X=0.4, with 2 tracks wide
        warr3 = self.add_wires(hm_layer, 4, 200, 400, width=2, unit_mode=True)

        # add 3 parallel vertical wires starting on track 6 and use every other track
        warr4 = self.add_wires(vm_layer, 6, 100, 400, num=3, pitch=2, unit_mode=True)
        print(warr4)

        # create a TrackID object representing a vertical track
        tid = TrackID(vm_layer, 3, width=2, num=1, pitch=0)
        # connect horizontal wires to the vertical track
        warr5 = self.connect_to_tracks([warr1, warr3], tid)
        print(warr5)

        # add a pin on a WireArray
        self.add_pin('pin1', warr1)
        # add a pin, but make label different than net name.  Useful for LVS connect
        self.add_pin('pin2', warr2, label='pin2:')
        # add_pin also works for WireArray representing multiple wires
        self.add_pin('pin3', warr4)
        # add a pin (so it is visible in BAG), but do not create the actual layout
        # in OA.  This is useful for hiding pins on lower levels of hierarchy.
        self.add_pin('pin4', warr3, show=False)

        # set the size of this template
        top_layer = vm_layer
        num_h_tracks = 6
        num_v_tracks = 11
        # size is 3-element tuple of top layer ID, number of top
        # vertical tracks, and number of top horizontal tracks
        self.size = top_layer, num_v_tracks, num_h_tracks
        # print bounding box of this template
        print(self.bound_box)
        # add a M7 rectangle to visualize bounding box in layout
        self.add_rect('M7', self.bound_box)


class AmpCS(AnalogBase):
    """A common source amplifier."""
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w_dict='width dictionary.',
            intent_dict='intent dictionary.',
            fg_dict='number of fingers dictionary.',
            ndum='number of dummies on each side.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            show_pins='True to draw pin geometries.',
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        lch = self.params['lch']
        w_dict = self.params['w_dict']
        intent_dict = self.params['intent_dict']
        fg_dict = self.params['fg_dict']
        ndum = self.params['ndum']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        show_pins = self.params['show_pins']

        fg_amp = fg_dict['amp']
        fg_load = fg_dict['load']

        if fg_load % 2 != 0 or fg_amp % 2 != 0:
            raise ValueError('fg_load=%d and fg_amp=%d must all be even.' % (fg_load, fg_amp))

        # compute total number of fingers in each row
        fg_half_pmos = fg_load // 2
        fg_half_nmos = fg_amp // 2
        fg_half = max(fg_half_pmos, fg_half_nmos)
        fg_tot = (fg_half + ndum) * 2

        # specify width/threshold of each row
        nw_list = [w_dict['amp']]
        pw_list = [w_dict['load']]
        nth_list = [intent_dict['amp']]
        pth_list = [intent_dict['load']]

        # specify number of horizontal tracks for each row
        ng_tracks = [1]  # input track
        nds_tracks = [1]  # one track for space
        pds_tracks = [1]  # output track
        pg_tracks = [1]  # bias track

        # specify row orientations
        n_orient = ['R0']  # gate connection on bottom
        p_orient = ['MX']  # gate connection on top

        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, pw_list, pth_list,
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                       n_orientations=n_orient, p_orientations=p_orient,
                       )

        # figure out if output connects to drain or source of nmos
        if (fg_amp - fg_load) % 4 == 0:
            s_net, d_net = '', 'vout'
            aout, aoutb, nsdir, nddir = 'd', 's', 0, 2
        else:
            s_net, d_net = 'vout', ''
            aout, aoutb, nsdir, nddir = 's', 'd', 2, 0

        # create transistor connections
        load_col = ndum + fg_half - fg_half_pmos
        amp_col = ndum + fg_half - fg_half_nmos
        amp_ports = self.draw_mos_conn('nch', 0, amp_col, fg_amp, nsdir, nddir,
                                       s_net=s_net, d_net=d_net)
        load_ports = self.draw_mos_conn('pch', 0, load_col, fg_load, 2, 0,
                                        s_net='', d_net='vout')
        # amp_ports/load_ports are dictionaries of WireArrays representing
        # transistor ports.
        print(amp_ports)
        print(amp_ports['g'])

        # create TrackID from relative track index
        vin_tid = self.make_track_id('nch', 0, 'g', 0)
        vbias_tid = self.make_track_id('pch', 0, 'g', 0)
        # can also convert from relative to absolute track index
        print(self.get_track_index('nch', 0, 'g', 0))
        # get output track index, put it in the middle
        vout_bot = self.get_track_index('nch', 0, 'ds', 0)
        vout_top = self.get_track_index('pch', 0, 'ds', 0)
        vout_index = self.grid.get_middle_track(vout_bot, vout_top, round_up=True)
        vout_tid = TrackID(self.mos_conn_layer + 1, vout_index)

        vin_warr = self.connect_to_tracks(amp_ports['g'], vin_tid)
        vout_warr = self.connect_to_tracks([amp_ports[aout], load_ports['d']], vout_tid)
        vbias_warr = self.connect_to_tracks(load_ports['g'], vbias_tid)
        self.connect_to_substrate('ptap', amp_ports[aoutb])
        self.connect_to_substrate('ntap', load_ports['s'])

        vss_warrs, vdd_warrs = self.fill_dummy()

        self.add_pin('VSS', vss_warrs, show=show_pins)
        self.add_pin('VDD', vdd_warrs, show=show_pins)
        self.add_pin('vin', vin_warr, show=show_pins)
        self.add_pin('vout', vout_warr, show=show_pins)
        self.add_pin('vbias', vbias_warr, show=show_pins)

        # compute schematic parameters
        self._sch_params = dict(
            lch=lch,
            w_dict=w_dict,
            intent_dict=intent_dict,
            fg_dict=fg_dict,
            dum_info=self.get_sch_dummy_info(),
        )


class AmpSF(AnalogBase):
    """A template of a single transistor with dummies.

    This class is mainly used for transistor characterization or
    design exploration with config views.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w_dict='width dictionary.',
            intent_dict='intent dictionary.',
            fg_dict='number of fingers dictionary.',
            ndum='number of dummies on each side.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            show_pins='True to draw pin geometries.',
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        lch = self.params['lch']
        w_dict = self.params['w_dict']
        intent_dict = self.params['intent_dict']
        fg_dict = self.params['fg_dict']
        ndum = self.params['ndum']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        show_pins = self.params['show_pins']

        fg_amp = fg_dict['amp']
        fg_bias = fg_dict['bias']

        if fg_bias % 2 != 0 or fg_amp % 2 != 0:
            raise ValueError('fg_bias=%d and fg_amp=%d must all be even.' % (fg_bias, fg_amp))

        fg_half_bias = fg_bias // 2
        fg_half_amp = fg_amp // 2
        fg_half = max(fg_half_bias, fg_half_amp)
        fg_tot = (fg_half + ndum) * 2

        nw_list = [w_dict['bias'], w_dict['amp']]
        nth_list = [intent_dict['bias'], intent_dict['amp']]
        ng_tracks = [1, 3]
        nds_tracks = [1, 1]

        n_orient = ['R0', 'MX']

        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, [], [],
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=[], pds_tracks=[],
                       n_orientations=n_orient,
                       )

        if (fg_amp - fg_bias) % 4 == 0:
            s_net, d_net = '', 'vout'
            aout, aoutb, nsdir, nddir = 'd', 's', 2, 0
        else:
            s_net, d_net = 'vout', ''
            aout, aoutb, nsdir, nddir = 's', 'd', 0, 2

        # TODO: compute bias_col and amp_col
        bias_col = amp_col = 0

        amp_ports = self.draw_mos_conn('nch', 1, amp_col, fg_amp, nsdir, nddir,
                                       s_net=s_net, d_net=d_net)
        bias_ports = self.draw_mos_conn('nch', 0, bias_col, fg_bias, 0, 2,
                                        s_net='', d_net='vout')

        # TODO: get TrackIDs for horizontal tracks
        # The following are related code copied and pasted from AmpCS
        # for reference
        # vin_tid = self.make_track_id('nch', 0, 'g', 0)
        # vout_tid = self.make_track_id('pch', 0, 'ds', 0)
        # vbias_tid = self.make_track_id('pch', 0, 'g', 0)
        vdd_tid = vin_tid = vout_tid = vbias_tid = None

        if vdd_tid is None:
            return

        # uncomment to visualize track location
        # hm_layer = self.mos_conn_layer + 1
        # xl = self.bound_box.left_unit
        # xr = self.bound_box.right_unit
        # self.add_wires(hm_layer, vdd_tid.base_index, xl, xr, unit_mode=True)
        # self.add_wires(hm_layer, vin_tid.base_index, xl, xr, unit_mode=True)
        # self.add_wires(hm_layer, vout_tid.base_index, xl, xr, unit_mode=True)
        # self.add_wires(hm_layer, vbias_tid.base_index, xl, xr, unit_mode=True)
        
        # TODO: connect transistors to horizontal tracks
        # The following are related code copied and pasted from AmpCS
        # for reference
        # vin_warr = self.connect_to_tracks(amp_ports['g'], vin_tid)
        # vout_warr = self.connect_to_tracks([amp_ports[aout], load_ports['d']], vout_tid)
        # vbias_warr = self.connect_to_tracks(load_ports['g'], vbias_tid)
        vin_warr = vout_warr = vbias_warr = vdd_warr = None

        if vin_warr is None:
            return

        self.connect_to_substrate('ptap', bias_ports['s'])

        vss_warrs, _ = self.fill_dummy()

        self.add_pin('VSS', vss_warrs, show=show_pins)
        # TODO: add pins

        # set schematic parameters
        self._sch_params = dict(
            lch=lch,
            w_dict=w_dict,
            intent_dict=intent_dict,
            fg_dict=fg_dict,
            dum_info=self.get_sch_dummy_info(),
        )


class AmpSFSoln(AnalogBase):
    """A template of a single transistor with dummies.

    This class is mainly used for transistor characterization or
    design exploration with config views.

    Parameters
    ----------
    temp_db : :class:`bag.layout.template.TemplateDB`
            the template database.
    lib_name : str
        the layout library name.
    params : dict[str, any]
        the parameter values.
    used_names : set[str]
        a set of already used cell names.
    kwargs : dict[str, any]
        dictionary of optional parameters.  See documentation of
        :class:`bag.layout.template.TemplateBase` for details.
    """

    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            w_dict='width dictionary.',
            intent_dict='intent dictionary.',
            fg_dict='number of fingers dictionary.',
            ndum='number of dummies on each side.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            show_pins='True to draw pin geometries.',
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        lch = self.params['lch']
        w_dict = self.params['w_dict']
        intent_dict = self.params['intent_dict']
        fg_dict = self.params['fg_dict']
        ndum = self.params['ndum']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        show_pins = self.params['show_pins']

        fg_amp = fg_dict['amp']
        fg_bias = fg_dict['bias']

        if fg_bias % 2 != 0 or fg_amp % 2 != 0:
            raise ValueError('fg_bias=%d and fg_amp=%d must all be even.' % (fg_bias, fg_amp))

        fg_half_bias = fg_bias // 2
        fg_half_amp = fg_amp // 2
        fg_half = max(fg_half_bias, fg_half_amp)
        fg_tot = (fg_half + ndum) * 2

        nw_list = [w_dict['bias'], w_dict['amp']]
        nth_list = [intent_dict['bias'], intent_dict['amp']]

        ng_tracks = [1, 3]
        nds_tracks = [1, 1]

        n_orient = ['R0', 'MX']

        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, [], [],
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=[], pds_tracks=[],
                       n_orientations=n_orient,
                       )

        if (fg_amp - fg_bias) % 4 == 0:
            s_net, d_net = 'VDD', 'vout'
            aout, aoutb, nsdir, nddir = 'd', 's', 2, 0
        else:
            s_net, d_net = 'vout', 'VDD'
            aout, aoutb, nsdir, nddir = 's', 'd', 0, 2

        bias_col = ndum + fg_half - fg_half_bias
        amp_col = ndum + fg_half - fg_half_amp
        amp_ports = self.draw_mos_conn('nch', 1, amp_col, fg_amp, nsdir, nddir,
                                       s_net=s_net, d_net=d_net)
        bias_ports = self.draw_mos_conn('nch', 0, bias_col, fg_bias, 0, 2,
                                        s_net='', d_net='vout')

        vdd_tid = self.make_track_id('nch', 1, 'g', 0)
        vin_tid = self.make_track_id('nch', 1, 'g', 2)
        vout_tid = self.make_track_id('nch', 0, 'ds', 0)
        vbias_tid = self.make_track_id('nch', 0, 'g', 0)

        vin_warr = self.connect_to_tracks(amp_ports['g'], vin_tid)
        vout_warr = self.connect_to_tracks([amp_ports[aout], bias_ports['d']], vout_tid)
        vbias_warr = self.connect_to_tracks(bias_ports['g'], vbias_tid)
        vdd_warr = self.connect_to_tracks(amp_ports[aoutb], vdd_tid)
        self.connect_to_substrate('ptap', bias_ports['s'])

        vss_warrs, _ = self.fill_dummy()

        self.add_pin('VSS', vss_warrs, show=show_pins)
        self.add_pin('VDD', vdd_warr, show=show_pins)
        self.add_pin('vin', vin_warr, show=show_pins)
        self.add_pin('vout', vout_warr, show=show_pins)
        self.add_pin('vbias', vbias_warr, show=show_pins)

        self._sch_params = dict(
            lch=lch,
            w_dict=w_dict,
            intent_dict=intent_dict,
            fg_dict=fg_dict,
            dum_info=self.get_sch_dummy_info(),
        )


class AmpChain(TemplateBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            cs_params='common source amplifier parameters.',
            sf_params='source follower parameters.',
            show_pins='True to draw pin geometries.',
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        # make copies of given dictionaries to avoid modifying external data.
        cs_params = self.params['cs_params'].copy()
        sf_params = self.params['sf_params'].copy()
        show_pins = self.params['show_pins']

        # disable pins in subcells
        cs_params['show_pins'] = False
        sf_params['show_pins'] = False

        # create layout masters for subcells we will add later
        cs_master = self.new_template(params=cs_params, temp_cls=AmpCS)
        # TODO: create sf_master
        sf_master = None

        if sf_master is None:
            return

        # add subcell instances
        cs_inst = self.add_instance(cs_master, 'XCS')
        # add source follower to the right of common source
        x0 = cs_inst.bound_box.right_unit
        sf_inst = self.add_instance(sf_master, 'XSF', loc=(x0, 0), unit_mode=True)

        # get VSS wires from AmpCS/AmpSF
        cs_vss_warr = cs_inst.get_all_port_pins('VSS')[0]
        sf_vss_warrs = sf_inst.get_all_port_pins('VSS')
        # only connect bottom VSS wire of source follower
        if len(sf_vss_warrs) < 2 or sf_vss_warrs[0].track_id.base_index < sf_vss_warrs[1].track_id.base_index:
            sf_vss_warr = sf_vss_warrs[0]
        else:
            sf_vss_warr = sf_vss_warrs[1]

        # connect VSS of the two blocks together
        vss = self.connect_wires([cs_vss_warr, sf_vss_warr])[0]

        # get layer IDs from VSS wire
        hm_layer = vss.layer_id
        vm_layer = hm_layer + 1
        top_layer = vm_layer + 1

        # calculate template size
        tot_box = cs_inst.bound_box.merge(sf_inst.bound_box)
        self.set_size_from_bound_box(top_layer, tot_box, round_up=True)

        # get subcell ports as WireArrays so we can connect them
        vmid0 = cs_inst.get_all_port_pins('vout')[0]
        vmid1 = sf_inst.get_all_port_pins('vin')[0]
        vdd0 = cs_inst.get_all_port_pins('VDD')[0]
        vdd1 = sf_inst.get_all_port_pins('VDD')[0]

        # get vertical VDD TrackIDs
        vdd0_tid = TrackID(vm_layer, self.grid.coord_to_nearest_track(vm_layer, vdd0.middle))
        vdd1_tid = TrackID(vm_layer, self.grid.coord_to_nearest_track(vm_layer, vdd1.middle))

        # connect VDD of each block to vertical M5
        vdd0 = self.connect_to_tracks(vdd0, vdd0_tid)
        vdd1 = self.connect_to_tracks(vdd1, vdd1_tid)
        # connect M5 VDD to top M6 horizontal track
        vdd_tidx = self.grid.get_num_tracks(self.size, top_layer) - 1
        vdd_tid = TrackID(top_layer, vdd_tidx)
        vdd = self.connect_to_tracks([vdd0, vdd1], vdd_tid)

        # TODO: connect vmid0 and vmid1 to vertical track in the middle of two templates
        # hint: use x0
        vmid = None

        if vmid is None:
            return

        # add pins on wires
        self.add_pin('vmid', vmid, show=show_pins)
        self.add_pin('VDD', vdd, show=show_pins)
        self.add_pin('VSS', vss, show=show_pins)
        # re-export pins on subcells.
        self.reexport(cs_inst.get_port('vin'), show=show_pins)
        self.reexport(cs_inst.get_port('vbias'), net_name='vb1', show=show_pins)
        # TODO: reexport vout and vbias of source follower
        # TODO: vbias should be renamed to vb2

        # compute schematic parameters.
        self._sch_params = dict(
            cs_params=cs_master.sch_params,
            sf_params=sf_master.sch_params,
        )


class AmpChainSoln(TemplateBase):
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            cs_params='common source amplifier parameters.',
            sf_params='source follower parameters.',
            show_pins='True to draw pin geometries.',
        )

    def draw_layout(self):
        """Draw the layout of a transistor for characterization.
        """

        cs_params = self.params['cs_params'].copy()
        sf_params = self.params['sf_params'].copy()
        show_pins = self.params['show_pins']

        cs_params['show_pins'] = False
        sf_params['show_pins'] = False

        # create layout masters for subcells we will add later
        cs_master = self.new_template(params=cs_params, temp_cls=AmpCS)
        sf_master = self.new_template(params=sf_params, temp_cls=AmpSFSoln)

        # add subcell instances
        cs_inst = self.add_instance(cs_master, 'XCS')
        # add source follower to the right of common source
        x0 = cs_inst.bound_box.right_unit
        sf_inst = self.add_instance(sf_master, 'XSF', loc=(x0, 0), unit_mode=True)

        # get VSS wires from AmpCS/AmpSF
        cs_vss_warr = cs_inst.get_all_port_pins('VSS')[0]
        sf_vss_warrs = sf_inst.get_all_port_pins('VSS')
        # only connect bottom VSS wire of source follower
        if len(sf_vss_warrs) < 2 or sf_vss_warrs[0].track_id.base_index < sf_vss_warrs[1].track_id.base_index:
            sf_vss_warr = sf_vss_warrs[0]
        else:
            sf_vss_warr = sf_vss_warrs[1]

        # connect VSS of the two blocks together
        vss = self.connect_wires([cs_vss_warr, sf_vss_warr])[0]

        # get layer IDs from VSS wire
        hm_layer = vss.layer_id
        vm_layer = hm_layer + 1
        top_layer = vm_layer + 1

        # calculate template size
        tot_box = cs_inst.bound_box.merge(sf_inst.bound_box)
        self.set_size_from_bound_box(top_layer, tot_box, round_up=True)

        # get subcell ports as WireArrays so we can connect them
        vmid0 = cs_inst.get_all_port_pins('vout')[0]
        vmid1 = sf_inst.get_all_port_pins('vin')[0]
        vdd0 = cs_inst.get_all_port_pins('VDD')[0]
        vdd1 = sf_inst.get_all_port_pins('VDD')[0]

        # get vertical VDD TrackIDs
        top_w = self.grid.get_track_width(top_layer, 1, unit_mode=True)
        vm_w_vdd = self.grid.get_min_track_width(vm_layer, top_w=top_w, unit_mode=True)
        vdd0_tid = TrackID(vm_layer, self.grid.coord_to_nearest_track(vm_layer, vdd0.middle),
                           width=vm_w_vdd)
        vdd1_tid = TrackID(vm_layer, self.grid.coord_to_nearest_track(vm_layer, vdd1.middle),
                           width=vm_w_vdd)

        # connect VDD of each block to vertical M5
        vdd0 = self.connect_to_tracks(vdd0, vdd0_tid)
        vdd1 = self.connect_to_tracks(vdd1, vdd1_tid)
        # connect M5 VDD to top M6 horizontal track
        vdd_tidx = self.grid.get_num_tracks(self.size, top_layer) - 1
        vdd_tid = TrackID(top_layer, vdd_tidx)
        vdd = self.connect_to_tracks([vdd0, vdd1], vdd_tid)

        # connect vmid using vertical track in the middle of the two templates
        mid_tid = TrackID(vm_layer, self.grid.coord_to_nearest_track(vm_layer, x0, unit_mode=True))
        vmid = self.connect_to_tracks([vmid0, vmid1], mid_tid)

        # add pins on wires
        self.add_pin('vmid', vmid, show=show_pins)
        self.add_pin('VDD', vdd, show=show_pins)
        self.add_pin('VSS', vss, show=show_pins)
        # re-export pins on subcells.
        self.reexport(cs_inst.get_port('vin'), show=show_pins)
        self.reexport(cs_inst.get_port('vbias'), net_name='vb1', show=show_pins)
        self.reexport(sf_inst.get_port('vout'), show=show_pins)
        self.reexport(sf_inst.get_port('vbias'), net_name='vb2', show=show_pins)

        # compute schematic parameters.
        self._sch_params = dict(
            cs_params=cs_master.sch_params,
            sf_params=sf_master.sch_params,
        )
