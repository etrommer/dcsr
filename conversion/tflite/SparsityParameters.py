# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SparsityParameters(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SparsityParameters()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSparsityParameters(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SparsityParametersBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # SparsityParameters
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SparsityParameters
    def TraversalOrder(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SparsityParameters
    def TraversalOrderAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SparsityParameters
    def TraversalOrderLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityParameters
    def TraversalOrderIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SparsityParameters
    def BlockMap(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SparsityParameters
    def BlockMapAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SparsityParameters
    def BlockMapLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityParameters
    def BlockMapIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # SparsityParameters
    def DimMetadata(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.DimensionMetadata import DimensionMetadata
            obj = DimensionMetadata()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SparsityParameters
    def DimMetadataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SparsityParameters
    def DimMetadataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # SparsityParameters
    def CompSparsity(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from tflite.CompressedSparsity import CompressedSparsity
            obj = CompressedSparsity()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def SparsityParametersStart(builder): builder.StartObject(4)
def Start(builder):
    return SparsityParametersStart(builder)
def SparsityParametersAddTraversalOrder(builder, traversalOrder): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(traversalOrder), 0)
def AddTraversalOrder(builder, traversalOrder):
    return SparsityParametersAddTraversalOrder(builder, traversalOrder)
def SparsityParametersStartTraversalOrderVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartTraversalOrderVector(builder, numElems):
    return SparsityParametersStartTraversalOrderVector(builder, numElems)
def SparsityParametersAddBlockMap(builder, blockMap): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(blockMap), 0)
def AddBlockMap(builder, blockMap):
    return SparsityParametersAddBlockMap(builder, blockMap)
def SparsityParametersStartBlockMapVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartBlockMapVector(builder, numElems):
    return SparsityParametersStartBlockMapVector(builder, numElems)
def SparsityParametersAddDimMetadata(builder, dimMetadata): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dimMetadata), 0)
def AddDimMetadata(builder, dimMetadata):
    return SparsityParametersAddDimMetadata(builder, dimMetadata)
def SparsityParametersStartDimMetadataVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartDimMetadataVector(builder, numElems):
    return SparsityParametersStartDimMetadataVector(builder, numElems)
def SparsityParametersAddCompSparsity(builder, compSparsity): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(compSparsity), 0)
def AddCompSparsity(builder, compSparsity):
    return SparsityParametersAddCompSparsity(builder, compSparsity)
def SparsityParametersEnd(builder): return builder.EndObject()
def End(builder):
    return SparsityParametersEnd(builder)
import tflite.CompressedSparsity
import tflite.DimensionMetadata
try:
    from typing import List, Optional
except:
    pass

class SparsityParametersT(object):

    # SparsityParametersT
    def __init__(self):
        self.traversalOrder = None  # type: List[int]
        self.blockMap = None  # type: List[int]
        self.dimMetadata = None  # type: List[tflite.DimensionMetadata.DimensionMetadataT]
        self.compSparsity = None  # type: Optional[tflite.CompressedSparsity.CompressedSparsityT]

    @classmethod
    def InitFromBuf(cls, buf, pos):
        sparsityParameters = SparsityParameters()
        sparsityParameters.Init(buf, pos)
        return cls.InitFromObj(sparsityParameters)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos+n)

    @classmethod
    def InitFromObj(cls, sparsityParameters):
        x = SparsityParametersT()
        x._UnPack(sparsityParameters)
        return x

    # SparsityParametersT
    def _UnPack(self, sparsityParameters):
        if sparsityParameters is None:
            return
        if not sparsityParameters.TraversalOrderIsNone():
            if np is None:
                self.traversalOrder = []
                for i in range(sparsityParameters.TraversalOrderLength()):
                    self.traversalOrder.append(sparsityParameters.TraversalOrder(i))
            else:
                self.traversalOrder = sparsityParameters.TraversalOrderAsNumpy()
        if not sparsityParameters.BlockMapIsNone():
            if np is None:
                self.blockMap = []
                for i in range(sparsityParameters.BlockMapLength()):
                    self.blockMap.append(sparsityParameters.BlockMap(i))
            else:
                self.blockMap = sparsityParameters.BlockMapAsNumpy()
        if not sparsityParameters.DimMetadataIsNone():
            self.dimMetadata = []
            for i in range(sparsityParameters.DimMetadataLength()):
                if sparsityParameters.DimMetadata(i) is None:
                    self.dimMetadata.append(None)
                else:
                    dimensionMetadata_ = tflite.DimensionMetadata.DimensionMetadataT.InitFromObj(sparsityParameters.DimMetadata(i))
                    self.dimMetadata.append(dimensionMetadata_)
        if sparsityParameters.CompSparsity() is not None:
            self.compSparsity = tflite.CompressedSparsity.CompressedSparsityT.InitFromObj(sparsityParameters.CompSparsity())

    # SparsityParametersT
    def Pack(self, builder):
        if self.traversalOrder is not None:
            if np is not None and type(self.traversalOrder) is np.ndarray:
                traversalOrder = builder.CreateNumpyVector(self.traversalOrder)
            else:
                SparsityParametersStartTraversalOrderVector(builder, len(self.traversalOrder))
                for i in reversed(range(len(self.traversalOrder))):
                    builder.PrependInt32(self.traversalOrder[i])
                traversalOrder = builder.EndVector()
        if self.blockMap is not None:
            if np is not None and type(self.blockMap) is np.ndarray:
                blockMap = builder.CreateNumpyVector(self.blockMap)
            else:
                SparsityParametersStartBlockMapVector(builder, len(self.blockMap))
                for i in reversed(range(len(self.blockMap))):
                    builder.PrependInt32(self.blockMap[i])
                blockMap = builder.EndVector()
        if self.dimMetadata is not None:
            dimMetadatalist = []
            for i in range(len(self.dimMetadata)):
                dimMetadatalist.append(self.dimMetadata[i].Pack(builder))
            SparsityParametersStartDimMetadataVector(builder, len(self.dimMetadata))
            for i in reversed(range(len(self.dimMetadata))):
                builder.PrependUOffsetTRelative(dimMetadatalist[i])
            dimMetadata = builder.EndVector()
        if self.compSparsity is not None:
            compSparsity = self.compSparsity.Pack(builder)
        SparsityParametersStart(builder)
        if self.traversalOrder is not None:
            SparsityParametersAddTraversalOrder(builder, traversalOrder)
        if self.blockMap is not None:
            SparsityParametersAddBlockMap(builder, blockMap)
        if self.dimMetadata is not None:
            SparsityParametersAddDimMetadata(builder, dimMetadata)
        if self.compSparsity is not None:
            SparsityParametersAddCompSparsity(builder, compSparsity)
        sparsityParameters = SparsityParametersEnd(builder)
        return sparsityParameters
