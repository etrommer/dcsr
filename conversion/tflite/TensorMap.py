# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TensorMap(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TensorMap()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTensorMap(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def TensorMapBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # TensorMap
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # TensorMap
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # TensorMap
    def TensorIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(2)
def TensorMapStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def TensorMapAddName(builder, name):
    """This method is deprecated. Please switch to AddName."""
    return AddName(builder, name)
def AddTensorIndex(builder, tensorIndex): builder.PrependUint32Slot(1, tensorIndex, 0)
def TensorMapAddTensorIndex(builder, tensorIndex):
    """This method is deprecated. Please switch to AddTensorIndex."""
    return AddTensorIndex(builder, tensorIndex)
def End(builder): return builder.EndObject()
def TensorMapEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class TensorMapT(object):

    # TensorMapT
    def __init__(self):
        self.name = None  # type: str
        self.tensorIndex = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        tensorMap = TensorMap()
        tensorMap.Init(buf, pos)
        return cls.InitFromObj(tensorMap)

    @classmethod
    def InitFromObj(cls, tensorMap):
        x = TensorMapT()
        x._UnPack(tensorMap)
        return x

    # TensorMapT
    def _UnPack(self, tensorMap):
        if tensorMap is None:
            return
        self.name = tensorMap.Name()
        self.tensorIndex = tensorMap.TensorIndex()

    # TensorMapT
    def Pack(self, builder):
        if self.name is not None:
            name = builder.CreateString(self.name)
        Start(builder)
        if self.name is not None:
            AddName(builder, name)
        AddTensorIndex(builder, self.tensorIndex)
        tensorMap = End(builder)
        return tensorMap