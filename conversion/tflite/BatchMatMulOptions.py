# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class BatchMatMulOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BatchMatMulOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsBatchMatMulOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def BatchMatMulOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # BatchMatMulOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BatchMatMulOptions
    def AdjX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # BatchMatMulOptions
    def AdjY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # BatchMatMulOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def Start(builder): builder.StartObject(3)
def BatchMatMulOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddAdjX(builder, adjX): builder.PrependBoolSlot(0, adjX, 0)
def BatchMatMulOptionsAddAdjX(builder, adjX):
    """This method is deprecated. Please switch to AddAdjX."""
    return AddAdjX(builder, adjX)
def AddAdjY(builder, adjY): builder.PrependBoolSlot(1, adjY, 0)
def BatchMatMulOptionsAddAdjY(builder, adjY):
    """This method is deprecated. Please switch to AddAdjY."""
    return AddAdjY(builder, adjY)
def AddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs): builder.PrependBoolSlot(2, asymmetricQuantizeInputs, 0)
def BatchMatMulOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    """This method is deprecated. Please switch to AddAsymmetricQuantizeInputs."""
    return AddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs)
def End(builder): return builder.EndObject()
def BatchMatMulOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class BatchMatMulOptionsT(object):

    # BatchMatMulOptionsT
    def __init__(self):
        self.adjX = False  # type: bool
        self.adjY = False  # type: bool
        self.asymmetricQuantizeInputs = False  # type: bool

    @classmethod
    def InitFromBuf(cls, buf, pos):
        batchMatMulOptions = BatchMatMulOptions()
        batchMatMulOptions.Init(buf, pos)
        return cls.InitFromObj(batchMatMulOptions)

    @classmethod
    def InitFromObj(cls, batchMatMulOptions):
        x = BatchMatMulOptionsT()
        x._UnPack(batchMatMulOptions)
        return x

    # BatchMatMulOptionsT
    def _UnPack(self, batchMatMulOptions):
        if batchMatMulOptions is None:
            return
        self.adjX = batchMatMulOptions.AdjX()
        self.adjY = batchMatMulOptions.AdjY()
        self.asymmetricQuantizeInputs = batchMatMulOptions.AsymmetricQuantizeInputs()

    # BatchMatMulOptionsT
    def Pack(self, builder):
        Start(builder)
        AddAdjX(builder, self.adjX)
        AddAdjY(builder, self.adjY)
        AddAsymmetricQuantizeInputs(builder, self.asymmetricQuantizeInputs)
        batchMatMulOptions = End(builder)
        return batchMatMulOptions
