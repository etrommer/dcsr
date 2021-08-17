# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RankOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RankOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRankOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def RankOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # RankOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def RankOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def RankOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class RankOptionsT(object):

    # RankOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        rankOptions = RankOptions()
        rankOptions.Init(buf, pos)
        return cls.InitFromObj(rankOptions)

    @classmethod
    def InitFromObj(cls, rankOptions):
        x = RankOptionsT()
        x._UnPack(rankOptions)
        return x

    # RankOptionsT
    def _UnPack(self, rankOptions):
        if rankOptions is None:
            return

    # RankOptionsT
    def Pack(self, builder):
        Start(builder)
        rankOptions = End(builder)
        return rankOptions
