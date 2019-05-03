# -*- coding: utf-8 -*-

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info',
                                                                   'amp_sf_soln.yaml'))


# noinspection PyPep8Naming
class demo_templates__amp_sf_soln(Module):
    """Module for library demo_templates cell amp_sf_soln.

    Fill in high level description here.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict(
            lch='channel length in meters.',
            w_dict='Dictionary of transistor widths.',
            intent_dict='Dictionary of transistor threshold flavors.',
            fg_dict='Dictionary of transistor number of fingers.',
            dum_info='Dummy information data structure',
        )

    def design(self, lch, w_dict, intent_dict, fg_dict, dum_info):
        w_amp = w_dict['amp']
        w_bias = w_dict['bias']
        intent_amp = intent_dict['amp']
        intent_bias = intent_dict['bias']
        fg_amp = fg_dict['amp']
        fg_bias = fg_dict['bias']

        self.instances['XAMP'].design(w=w_amp, l=lch, intent=intent_amp, nf=fg_amp)
        self.instances['XBIAS'].design(w=w_bias, l=lch, intent=intent_bias, nf=fg_bias)

        # handle dummy transistors
        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
