# -*- coding: utf-8 -*-

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info',
                                                                   'amp_chain_soln.yaml'))


# noinspection PyPep8Naming
class demo_templates__amp_chain_soln(Module):
    """Module for library demo_templates cell amp_chain_soln.

    Fill in high level description here.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        return dict(
            cs_params='common-source amplifier parameters.',
            sf_params='source-follower amplifier parameters.',
        )

    def design(self, cs_params, sf_params):

        self.instances['XCS'].design(**cs_params)
        self.instances['XSF'].design(**sf_params)
