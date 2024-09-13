# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlashBenchData

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class AttentionSolution(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AttentionSolution()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsAttentionSolution(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # AttentionSolution
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # AttentionSolution
    def HeadDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionSolution
    def TileM(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionSolution
    def TileN(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionSolution
    def NumWaves(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionSolution
    def GridType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # AttentionSolution
    def BlanceType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # AttentionSolution
    def OpType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def AttentionSolutionStart(builder):
    builder.StartObject(7)

def Start(builder):
    AttentionSolutionStart(builder)

def AttentionSolutionAddHeadDim(builder, headDim):
    builder.PrependInt32Slot(0, headDim, 0)

def AddHeadDim(builder, headDim):
    AttentionSolutionAddHeadDim(builder, headDim)

def AttentionSolutionAddTileM(builder, tileM):
    builder.PrependInt32Slot(1, tileM, 0)

def AddTileM(builder, tileM):
    AttentionSolutionAddTileM(builder, tileM)

def AttentionSolutionAddTileN(builder, tileN):
    builder.PrependInt32Slot(2, tileN, 0)

def AddTileN(builder, tileN):
    AttentionSolutionAddTileN(builder, tileN)

def AttentionSolutionAddNumWaves(builder, numWaves):
    builder.PrependInt32Slot(3, numWaves, 0)

def AddNumWaves(builder, numWaves):
    AttentionSolutionAddNumWaves(builder, numWaves)

def AttentionSolutionAddGridType(builder, gridType):
    builder.PrependInt8Slot(4, gridType, 0)

def AddGridType(builder, gridType):
    AttentionSolutionAddGridType(builder, gridType)

def AttentionSolutionAddBlanceType(builder, blanceType):
    builder.PrependInt8Slot(5, blanceType, 0)

def AddBlanceType(builder, blanceType):
    AttentionSolutionAddBlanceType(builder, blanceType)

def AttentionSolutionAddOpType(builder, opType):
    builder.PrependInt8Slot(6, opType, 0)

def AddOpType(builder, opType):
    AttentionSolutionAddOpType(builder, opType)

def AttentionSolutionEnd(builder):
    return builder.EndObject()

def End(builder):
    return AttentionSolutionEnd(builder)
