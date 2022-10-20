# Copyright (c) Facebook, Inc. and its affiliates.
from iopath.common.file_io import PathHandler
from iopath.common.file_io import PathManager as PathManagerBase

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()
