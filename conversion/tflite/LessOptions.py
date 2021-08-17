# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LessOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LessOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsLessOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def LessOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # LessOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def LessOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def LessOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class LessOptionsT(object):

    # LessOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        lessOptions = LessOptions()
        lessOptions.Init(buf, pos)
        return cls.InitFromObj(lessOptions)

    @classmethod
    def InitFromObj(cls, lessOptions):
        x = LessOptionsT()
        x._UnPack(lessOptions)
        return x

    # LessOptionsT
    def _UnPack(self, lessOptions):
        if lessOptions is None:
            return

    # LessOptionsT
    def Pack(self, builder):
        Start(builder)
        lessOptions = End(builder)
        return lessOptions