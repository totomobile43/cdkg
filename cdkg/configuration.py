import os
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class Configuration(object):

    class __Configuration:

        def __init__(self):
            # Merge with CDKG Configuration.
            self._conf = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cdkg.yaml'))
            self._gpu_available = torch.cuda.is_available()

        def update_conf(self, conf_obj):
            """Update the configuration with another configuration file or another OmegaConf configuration object."""
            if isinstance(conf_obj, str):
                conf_obj = OmegaConf.load(conf_obj)
            else:
                assert isinstance(conf_obj, DictConfig)
            self._conf.merge_with(conf_obj)

        def __getattr__(self, item):
            if hasattr(self._conf, item):
                # Some attributes of the config are converted to torch objects automatically.
                if item == 'device':
                    return torch.device(self._conf.get('device', 'cuda:0') if self._gpu_available else 'cpu')
                elif item == 'f_precision':
                    return getattr(torch, 'float{}'.format(self._conf.get('f_precision', 32)))
                elif item == 'i_precision':
                    return getattr(torch, 'int{}'.format(self._conf.get('i_precision', 64)))
                else:
                    return getattr(self._conf, item)
            else:
                # Default behavior.
                return self.__getattribute__(item)

        @property
        def conf(self):
            return self._conf

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Configuration.instance:
            Configuration.instance = Configuration.__Configuration()
        return Configuration.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)

CONFIG = Configuration()
