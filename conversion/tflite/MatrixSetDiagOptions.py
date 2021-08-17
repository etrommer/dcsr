# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class MatrixSetDiagOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MatrixSetDiagOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMatrixSetDiagOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def MatrixSetDiagOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # MatrixSetDiagOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def MatrixSetDiagOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def MatrixSetDiagOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)

class MatrixSetDiagOptionsT(object):

    # MatrixSetDiagOptionsT
    def __init__(self):
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        matrixSetDiagOptions = MatrixSetDiagOptions()
        matrixSetDiagOptions.Init(buf, pos)
        return cls.InitFromObj(matrixSetDiagOptions)

    @classmethod
    def InitFromObj(cls, matrixSetDiagOptions):
        x = MatrixSetDiagOptionsT()
        x._UnPack(matrixSetDiagOptions)
        return x

    # MatrixSetDiagOptionsT
    def _UnPack(self, matrixSetDiagOptions):
        if matrixSetDiagOptions is None:
            return

    # MatrixSetDiagOptionsT
    def Pack(self, builder):
        Start(builder)
        matrixSetDiagOptions = End(builder)
        return matrixSetDiagOptions
