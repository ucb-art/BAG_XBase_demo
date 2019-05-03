# -*- coding: utf-8 -*-

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'amp_cs.yaml'))


# noinspection PyPep8Naming
class demo_templates__amp_cs(Module):

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent,
                        prj=prj, **kwargs)

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
        # populate self.parameters dictionary
        wp = w_dict['load']
        wn = w_dict['amp']
        intentp = intent_dict['load']
        intentn = intent_dict['amp']

        fg_amp = fg_dict['amp']
        fg_load = fg_dict['load']

        # set transistor parameters
        self.instances['XP'].design(w=wp, l=lch, intent=intentp, nf=fg_load)
        self.instances['XN'].design(w=wn, l=lch, intent=intentn, nf=fg_amp)

        # handle dummy transistors
        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
