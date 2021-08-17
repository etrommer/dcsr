# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class CompressedSparsity(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CompressedSparsity()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsCompressedSparsity(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def CompressedSparsityBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # CompressedSparsity
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CompressedSparsity
    def RowOffsets(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # CompressedSparsity
    def RowOffsetsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # CompressedSparsity
    def RowOffsetsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CompressedSparsity
    def RowOffsetsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # CompressedSparsity
    def DeltaIndices(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # CompressedSparsity
    def DeltaIndicesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # CompressedSparsity
    def DeltaIndicesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CompressedSparsity
    def DeltaIndicesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # CompressedSparsity
    def GroupMinimums(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # CompressedSparsity
    def GroupMinimumsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    # CompressedSparsity
    def GroupMinimumsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CompressedSparsity
    def GroupMinimumsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # CompressedSparsity
    def Bitmaps(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # CompressedSparsity
    def BitmapsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # CompressedSparsity
    def BitmapsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CompressedSparsity
    def BitmapsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # CompressedSparsity
    def Bitmasks(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # CompressedSparsity
    def BitmasksAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint16Flags, o)
        return 0

    # CompressedSparsity
    def BitmasksLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # CompressedSparsity
    def BitmasksIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # CompressedSparsity
    def Nnze(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(6)
def CompressedSparsityStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddRowOffsets(builder, rowOffsets): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(rowOffsets), 0)
def CompressedSparsityAddRowOffsets(builder, rowOffsets):
    """This method is deprecated. Please switch to AddRowOffsets."""
    return AddRowOffsets(builder, rowOffsets)
def StartRowOffsetsVector(builder, numElems): return builder.StartVector(2, numElems, 2)
def CompressedSparsityStartRowOffsetsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartRowOffsetsVector(builder, numElems)
def AddDeltaIndices(builder, deltaIndices): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(deltaIndices), 0)
def CompressedSparsityAddDeltaIndices(builder, deltaIndices):
    """This method is deprecated. Please switch to AddDeltaIndices."""
    return AddDeltaIndices(builder, deltaIndices)
def StartDeltaIndicesVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def CompressedSparsityStartDeltaIndicesVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartDeltaIndicesVector(builder, numElems)
def AddGroupMinimums(builder, groupMinimums): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(groupMinimums), 0)
def CompressedSparsityAddGroupMinimums(builder, groupMinimums):
    """This method is deprecated. Please switch to AddGroupMinimums."""
    return AddGroupMinimums(builder, groupMinimums)
def StartGroupMinimumsVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def CompressedSparsityStartGroupMinimumsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartGroupMinimumsVector(builder, numElems)
def AddBitmaps(builder, bitmaps): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(bitmaps), 0)
def CompressedSparsityAddBitmaps(builder, bitmaps):
    """This method is deprecated. Please switch to AddBitmaps."""
    return AddBitmaps(builder, bitmaps)
def StartBitmapsVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def CompressedSparsityStartBitmapsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartBitmapsVector(builder, numElems)
def AddBitmasks(builder, bitmasks): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(bitmasks), 0)
def CompressedSparsityAddBitmasks(builder, bitmasks):
    """This method is deprecated. Please switch to AddBitmasks."""
    return AddBitmasks(builder, bitmasks)
def StartBitmasksVector(builder, numElems): return builder.StartVector(2, numElems, 2)
def CompressedSparsityStartBitmasksVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartBitmasksVector(builder, numElems)
def AddNnze(builder, nnze): builder.PrependUint32Slot(5, nnze, 0)
def CompressedSparsityAddNnze(builder, nnze):
    """This method is deprecated. Please switch to AddNnze."""
    return AddNnze(builder, nnze)
def End(builder): return builder.EndObject()
def CompressedSparsityEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
try:
    from typing import List
except:
    pass

class CompressedSparsityT(object):

    # CompressedSparsityT
    def __init__(self):
        self.rowOffsets = None  # type: List[int]
        self.deltaIndices = None  # type: List[int]
        self.groupMinimums = None  # type: List[int]
        self.bitmaps = None  # type: List[int]
        self.bitmasks = None  # type: List[int]
        self.nnze = 0  # type: int

    @classmethod
    def InitFromBuf(cls, buf, pos):
        compressedSparsity = CompressedSparsity()
        compressedSparsity.Init(buf, pos)
        return cls.InitFromObj(compressedSparsity)

    @classmethod
    def InitFromObj(cls, compressedSparsity):
        x = CompressedSparsityT()
        x._UnPack(compressedSparsity)
        return x

    # CompressedSparsityT
    def _UnPack(self, compressedSparsity):
        if compressedSparsity is None:
            return
        if not compressedSparsity.RowOffsetsIsNone():
            if np is None:
                self.rowOffsets = []
                for i in range(compressedSparsity.RowOffsetsLength()):
                    self.rowOffsets.append(compressedSparsity.RowOffsets(i))
            else:
                self.rowOffsets = compressedSparsity.RowOffsetsAsNumpy()
        if not compressedSparsity.DeltaIndicesIsNone():
            if np is None:
                self.deltaIndices = []
                for i in range(compressedSparsity.DeltaIndicesLength()):
                    self.deltaIndices.append(compressedSparsity.DeltaIndices(i))
            else:
                self.deltaIndices = compressedSparsity.DeltaIndicesAsNumpy()
        if not compressedSparsity.GroupMinimumsIsNone():
            if np is None:
                self.groupMinimums = []
                for i in range(compressedSparsity.GroupMinimumsLength()):
                    self.groupMinimums.append(compressedSparsity.GroupMinimums(i))
            else:
                self.groupMinimums = compressedSparsity.GroupMinimumsAsNumpy()
        if not compressedSparsity.BitmapsIsNone():
            if np is None:
                self.bitmaps = []
                for i in range(compressedSparsity.BitmapsLength()):
                    self.bitmaps.append(compressedSparsity.Bitmaps(i))
            else:
                self.bitmaps = compressedSparsity.BitmapsAsNumpy()
        if not compressedSparsity.BitmasksIsNone():
            if np is None:
                self.bitmasks = []
                for i in range(compressedSparsity.BitmasksLength()):
                    self.bitmasks.append(compressedSparsity.Bitmasks(i))
            else:
                self.bitmasks = compressedSparsity.BitmasksAsNumpy()
        self.nnze = compressedSparsity.Nnze()

    # CompressedSparsityT
    def Pack(self, builder):
        if self.rowOffsets is not None:
            if np is not None and type(self.rowOffsets) is np.ndarray:
                rowOffsets = builder.CreateNumpyVector(self.rowOffsets)
            else:
                StartRowOffsetsVector(builder, len(self.rowOffsets))
                for i in reversed(range(len(self.rowOffsets))):
                    builder.PrependInt16(self.rowOffsets[i])
                rowOffsets = builder.EndVector()
        if self.deltaIndices is not None:
            if np is not None and type(self.deltaIndices) is np.ndarray:
                deltaIndices = builder.CreateNumpyVector(self.deltaIndices)
            else:
                StartDeltaIndicesVector(builder, len(self.deltaIndices))
                for i in reversed(range(len(self.deltaIndices))):
                    builder.PrependUint8(self.deltaIndices[i])
                deltaIndices = builder.EndVector()
        if self.groupMinimums is not None:
            if np is not None and type(self.groupMinimums) is np.ndarray:
                groupMinimums = builder.CreateNumpyVector(self.groupMinimums)
            else:
                StartGroupMinimumsVector(builder, len(self.groupMinimums))
                for i in reversed(range(len(self.groupMinimums))):
                    builder.PrependByte(self.groupMinimums[i])
                groupMinimums = builder.EndVector()
        if self.bitmaps is not None:
            if np is not None and type(self.bitmaps) is np.ndarray:
                bitmaps = builder.CreateNumpyVector(self.bitmaps)
            else:
                StartBitmapsVector(builder, len(self.bitmaps))
                for i in reversed(range(len(self.bitmaps))):
                    builder.PrependUint8(self.bitmaps[i])
                bitmaps = builder.EndVector()
        if self.bitmasks is not None:
            if np is not None and type(self.bitmasks) is np.ndarray:
                bitmasks = builder.CreateNumpyVector(self.bitmasks)
            else:
                StartBitmasksVector(builder, len(self.bitmasks))
                for i in reversed(range(len(self.bitmasks))):
                    builder.PrependUint16(self.bitmasks[i])
                bitmasks = builder.EndVector()
        Start(builder)
        if self.rowOffsets is not None:
            AddRowOffsets(builder, rowOffsets)
        if self.deltaIndices is not None:
            AddDeltaIndices(builder, deltaIndices)
        if self.groupMinimums is not None:
            AddGroupMinimums(builder, groupMinimums)
        if self.bitmaps is not None:
            AddBitmaps(builder, bitmaps)
        if self.bitmasks is not None:
            AddBitmasks(builder, bitmasks)
        AddNnze(builder, self.nnze)
        compressedSparsity = End(builder)
        return compressedSparsity
