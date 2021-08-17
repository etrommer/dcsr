# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SignatureDef(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SignatureDef()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSignatureDef(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SignatureDefBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # SignatureDef
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SignatureDef
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.TensorMap import TensorMap
            obj = TensorMap()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SignatureDef
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SignatureDef
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SignatureDef
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.TensorMap import TensorMap
            obj = TensorMap()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SignatureDef
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SignatureDef
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # SignatureDef
    def MethodName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # SignatureDef
    def Key(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def Start(builder): builder.StartObject(4)
def SignatureDefStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddInputs(builder, inputs): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)
def SignatureDefAddInputs(builder, inputs):
    """This method is deprecated. Please switch to AddInputs."""
    return AddInputs(builder, inputs)
def StartInputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SignatureDefStartInputsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartInputsVector(builder, numElems)
def AddOutputs(builder, outputs): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)
def SignatureDefAddOutputs(builder, outputs):
    """This method is deprecated. Please switch to AddOutputs."""
    return AddOutputs(builder, outputs)
def StartOutputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SignatureDefStartOutputsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartOutputsVector(builder, numElems)
def AddMethodName(builder, methodName): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(methodName), 0)
def SignatureDefAddMethodName(builder, methodName):
    """This method is deprecated. Please switch to AddMethodName."""
    return AddMethodName(builder, methodName)
def AddKey(builder, key): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(key), 0)
def SignatureDefAddKey(builder, key):
    """This method is deprecated. Please switch to AddKey."""
    return AddKey(builder, key)
def End(builder): return builder.EndObject()
def SignatureDefEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
import tflite.TensorMap
try:
    from typing import List
except:
    pass

class SignatureDefT(object):

    # SignatureDefT
    def __init__(self):
        self.inputs = None  # type: List[tflite.TensorMap.TensorMapT]
        self.outputs = None  # type: List[tflite.TensorMap.TensorMapT]
        self.methodName = None  # type: str
        self.key = None  # type: str

    @classmethod
    def InitFromBuf(cls, buf, pos):
        signatureDef = SignatureDef()
        signatureDef.Init(buf, pos)
        return cls.InitFromObj(signatureDef)

    @classmethod
    def InitFromObj(cls, signatureDef):
        x = SignatureDefT()
        x._UnPack(signatureDef)
        return x

    # SignatureDefT
    def _UnPack(self, signatureDef):
        if signatureDef is None:
            return
        if not signatureDef.InputsIsNone():
            self.inputs = []
            for i in range(signatureDef.InputsLength()):
                if signatureDef.Inputs(i) is None:
                    self.inputs.append(None)
                else:
                    tensorMap_ = tflite.TensorMap.TensorMapT.InitFromObj(signatureDef.Inputs(i))
                    self.inputs.append(tensorMap_)
        if not signatureDef.OutputsIsNone():
            self.outputs = []
            for i in range(signatureDef.OutputsLength()):
                if signatureDef.Outputs(i) is None:
                    self.outputs.append(None)
                else:
                    tensorMap_ = tflite.TensorMap.TensorMapT.InitFromObj(signatureDef.Outputs(i))
                    self.outputs.append(tensorMap_)
        self.methodName = signatureDef.MethodName()
        self.key = signatureDef.Key()

    # SignatureDefT
    def Pack(self, builder):
        if self.inputs is not None:
            inputslist = []
            for i in range(len(self.inputs)):
                inputslist.append(self.inputs[i].Pack(builder))
            StartInputsVector(builder, len(self.inputs))
            for i in reversed(range(len(self.inputs))):
                builder.PrependUOffsetTRelative(inputslist[i])
            inputs = builder.EndVector()
        if self.outputs is not None:
            outputslist = []
            for i in range(len(self.outputs)):
                outputslist.append(self.outputs[i].Pack(builder))
            StartOutputsVector(builder, len(self.outputs))
            for i in reversed(range(len(self.outputs))):
                builder.PrependUOffsetTRelative(outputslist[i])
            outputs = builder.EndVector()
        if self.methodName is not None:
            methodName = builder.CreateString(self.methodName)
        if self.key is not None:
            key = builder.CreateString(self.key)
        Start(builder)
        if self.inputs is not None:
            AddInputs(builder, inputs)
        if self.outputs is not None:
            AddOutputs(builder, outputs)
        if self.methodName is not None:
            AddMethodName(builder, methodName)
        if self.key is not None:
            AddKey(builder, key)
        signatureDef = End(builder)
        return signatureDef