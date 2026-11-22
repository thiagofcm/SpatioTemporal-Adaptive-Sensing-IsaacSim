"""Minimal stand-in for ROS 2 ament_index_python, used only to expand the TurtleBot4 xacro without ROS.
Resolves a package name to the directory of the matching package.xml inside the pinned clones in third_party/."""
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKGS = {}
for _repo in ("turtlebot4", "create3_sim"):
    for _dir, _subdirs, _files in os.walk(os.path.join(_ROOT, _repo)):
        if "package.xml" in _files:
            with open(os.path.join(_dir, "package.xml")) as _f:
                _m = re.search(r"<name>\s*([^<\s]+)\s*</name>", _f.read())
            if _m:
                _PKGS[_m.group(1)] = _dir


class PackageNotFoundError(KeyError):
    pass


def get_package_share_directory(name):
    if name not in _PKGS:
        raise PackageNotFoundError(name)
    return _PKGS[name]
