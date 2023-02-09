# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class GatherNdOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GatherNdOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsGatherNdOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def GatherNdOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # GatherNdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def GatherNdOptionsStart(builder): builder.StartObject(0)
def Start(builder):
    return GatherNdOptionsStart(builder)
def GatherNdOptionsEnd(builder): return builder.EndObject()
def End(builder):
    return GatherNdOptionsEnd(builder)

class GatherNdOptionsT(object):

    # GatherNdOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        gatherNdOptions = GatherNdOptions()
        gatherNdOptions.Init(buf, pos)
        return cls.InitFromObj(gatherNdOptions)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, gatherNdOptions):
        x = GatherNdOptionsT()
        x._UnPack(gatherNdOptions)
        return x

    # GatherNdOptionsT
    def _UnPack(self, gatherNdOptions):
        if gatherNdOptions is None:
            return

    # GatherNdOptionsT
    def Pack(self, builder):
        GatherNdOptionsStart(builder)
        gatherNdOptions = GatherNdOptionsEnd(builder)
        return gatherNdOptions
