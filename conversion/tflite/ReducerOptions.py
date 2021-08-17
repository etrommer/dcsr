# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReducerOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReducerOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReducerOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ReducerOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # ReducerOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReducerOptions
    def KeepDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def Start(builder): builder.StartObject(1)
def ReducerOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddKeepDims(builder, keepDims): builder.PrependBoolSlot(0, keepDims, 0)
def ReducerOptionsAddKeepDims(builder, keepDims):
    """This method is deprecated. Please switch to AddKeepDims."""
    return AddKeepDims(builder, keepDims)
def End(builder): return builder.EndObject()
def ReducerOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class ReducerOptionsT(object):

    # ReducerOptionsT
    def __init__(self):
        self.keepDims = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        reducerOptions = ReducerOptions()
        reducerOptions.Init(buf, pos)
        return cls.InitFromObj(reducerOptions)

    @classmethod
    def InitFromObj(cls, reducerOptions):
        x = ReducerOptionsT()
        x._UnPack(reducerOptions)
        return x

    # ReducerOptionsT
    def _UnPack(self, reducerOptions):
        if reducerOptions is None:
            return
        self.keepDims = reducerOptions.KeepDims()

    # ReducerOptionsT
    def Pack(self, builder):
        Start(builder)
        AddKeepDims(builder, self.keepDims)
        reducerOptions = End(builder)
        return reducerOptions
