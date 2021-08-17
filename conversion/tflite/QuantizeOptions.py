# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class QuantizeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = QuantizeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsQuantizeOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def QuantizeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # QuantizeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def QuantizeOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def QuantizeOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class QuantizeOptionsT(object):

    # QuantizeOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        quantizeOptions = QuantizeOptions()
        quantizeOptions.Init(buf, pos)
        return cls.InitFromObj(quantizeOptions)

    @classmethod
    def InitFromObj(cls, quantizeOptions):
        x = QuantizeOptionsT()
        x._UnPack(quantizeOptions)
        return x

    # QuantizeOptionsT
    def _UnPack(self, quantizeOptions):
        if quantizeOptions is None:
            return

    # QuantizeOptionsT
    def Pack(self, builder):
        Start(builder)
        quantizeOptions = End(builder)
        return quantizeOptions
