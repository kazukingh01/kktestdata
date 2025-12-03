class DatasetError(Exception):
    pass

class UnsupportedError(DatasetError):
    pass

class UnsupportedFormatError(UnsupportedError):
    pass

class MissingDependencyError(DatasetError):
    pass
