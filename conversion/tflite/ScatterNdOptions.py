# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ScatterNdOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ScatterNdOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsScatterNdOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ScatterNdOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ScatterNdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def ScatterNdOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def ScatterNdOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class ScatterNdOptionsT(object):

    # ScatterNdOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        scatterNdOptions = ScatterNdOptions()
        scatterNdOptions.Init(buf, pos)
        return cls.InitFromObj(scatterNdOptions)

    @classmethod
    def InitFromObj(cls, scatterNdOptions):
        x = ScatterNdOptionsT()
        x._UnPack(scatterNdOptions)
        return x

    # ScatterNdOptionsT
    def _UnPack(self, scatterNdOptions):
        if scatterNdOptions is None:
            return

    # ScatterNdOptionsT
    def Pack(self, builder):
        Start(builder)
        scatterNdOptions = End(builder)
        return scatterNdOptions
