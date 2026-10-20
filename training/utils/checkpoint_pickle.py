"""Read EVREAL's legacy ConfigParser checkpoint without global import aliases."""
import pathlib
import pickle
import os


class ConfigParser:
    def __getitem__(self, name):
        return self._config[name]


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'parse_config' and name == 'ConfigParser':
            return ConfigParser
        if module == 'pathlib' and name == 'PosixPath' and os.name == 'nt':
            return pathlib.WindowsPath
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()
