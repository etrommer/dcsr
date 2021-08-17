# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Conv3DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Conv3DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConv3DOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def Conv3DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Conv3DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Conv3DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideD(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Conv3DOptions
    def DilationDFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Conv3DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

def Start(builder): builder.StartObject(8)
def Conv3DOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def Conv3DOptionsAddPadding(builder, padding):
    """This method is deprecated. Please switch to AddPadding."""
    return AddPadding(builder, padding)
def AddStrideD(builder, strideD): builder.PrependInt32Slot(1, strideD, 0)
def Conv3DOptionsAddStrideD(builder, strideD):
    """This method is deprecated. Please switch to AddStrideD."""
    return AddStrideD(builder, strideD)
def AddStrideW(builder, strideW): builder.PrependInt32Slot(2, strideW, 0)
def Conv3DOptionsAddStrideW(builder, strideW):
    """This method is deprecated. Please switch to AddStrideW."""
    return AddStrideW(builder, strideW)
def AddStrideH(builder, strideH): builder.PrependInt32Slot(3, strideH, 0)
def Conv3DOptionsAddStrideH(builder, strideH):
    """This method is deprecated. Please switch to AddStrideH."""
    return AddStrideH(builder, strideH)
def AddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(4, fusedActivationFunction, 0)
def Conv3DOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    """This method is deprecated. Please switch to AddFusedActivationFunction."""
    return AddFusedActivationFunction(builder, fusedActivationFunction)
def AddDilationDFactor(builder, dilationDFactor): builder.PrependInt32Slot(5, dilationDFactor, 1)
def Conv3DOptionsAddDilationDFactor(builder, dilationDFactor):
    """This method is deprecated. Please switch to AddDilationDFactor."""
    return AddDilationDFactor(builder, dilationDFactor)
def AddDilationWFactor(builder, dilationWFactor): builder.PrependInt32Slot(6, dilationWFactor, 1)
def Conv3DOptionsAddDilationWFactor(builder, dilationWFactor):
    """This method is deprecated. Please switch to AddDilationWFactor."""
    return AddDilationWFactor(builder, dilationWFactor)
def AddDilationHFactor(builder, dilationHFactor): builder.PrependInt32Slot(7, dilationHFactor, 1)
def Conv3DOptionsAddDilationHFactor(builder, dilationHFactor):
    """This method is deprecated. Please switch to AddDilationHFactor."""
    return AddDilationHFactor(builder, dilationHFactor)
def End(builder): return builder.EndObject()
def Conv3DOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class Conv3DOptionsT(object):

    # Conv3DOptionsT
    def __init__(self):
        self.padding = 0  # type: int
        self.strideD = 0  # type: int
        self.strideW = 0  # type: int
        self.strideH = 0  # type: int
        self.fusedActivationFunction = 0  # type: int
        self.dilationDFactor = 1  # type: int
        self.dilationWFactor = 1  # type: int
        self.dilationHFactor = 1  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        conv3DOptions = Conv3DOptions()
        conv3DOptions.Init(buf, pos)
        return cls.InitFromObj(conv3DOptions)

    @classmethod
    def InitFromObj(cls, conv3DOptions):
        x = Conv3DOptionsT()
        x._UnPack(conv3DOptions)
        return x

    # Conv3DOptionsT
    def _UnPack(self, conv3DOptions):
        if conv3DOptions is None:
            return
        self.padding = conv3DOptions.Padding()
        self.strideD = conv3DOptions.StrideD()
        self.strideW = conv3DOptions.StrideW()
        self.strideH = conv3DOptions.StrideH()
        self.fusedActivationFunction = conv3DOptions.FusedActivationFunction()
        self.dilationDFactor = conv3DOptions.DilationDFactor()
        self.dilationWFactor = conv3DOptions.DilationWFactor()
        self.dilationHFactor = conv3DOptions.DilationHFactor()

    # Conv3DOptionsT
    def Pack(self, builder):
        Start(builder)
        AddPadding(builder, self.padding)
        AddStrideD(builder, self.strideD)
        AddStrideW(builder, self.strideW)
        AddStrideH(builder, self.strideH)
        AddFusedActivationFunction(builder, self.fusedActivationFunction)
        AddDilationDFactor(builder, self.dilationDFactor)
        AddDilationWFactor(builder, self.dilationWFactor)
        AddDilationHFactor(builder, self.dilationHFactor)
        conv3DOptions = End(builder)
        return conv3DOptions
