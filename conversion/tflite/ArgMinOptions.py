# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ArgMinOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArgMinOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsArgMinOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ArgMinOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ArgMinOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ArgMinOptions
    def OutputType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(1)
def ArgMinOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddOutputType(builder, outputType): builder.PrependInt8Slot(0, outputType, 0)
def ArgMinOptionsAddOutputType(builder, outputType):
    """This method is deprecated. Please switch to AddOutputType."""
    return AddOutputType(builder, outputType)
def End(builder): return builder.EndObject()
def ArgMinOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class ArgMinOptionsT(object):

    # ArgMinOptionsT
    def __init__(self):
        self.outputType = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        argMinOptions = ArgMinOptions()
        argMinOptions.Init(buf, pos)
        return cls.InitFromObj(argMinOptions)

    @classmethod
    def InitFromObj(cls, argMinOptions):
        x = ArgMinOptionsT()
        x._UnPack(argMinOptions)
        return x

    # ArgMinOptionsT
    def _UnPack(self, argMinOptions):
        if argMinOptions is None:
            return
        self.outputType = argMinOptions.OutputType()

    # ArgMinOptionsT
    def Pack(self, builder):
        Start(builder)
        AddOutputType(builder, self.outputType)
        argMinOptions = End(builder)
        return argMinOptions
