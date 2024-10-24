# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlashBenchData

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class AttentionProblem(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AttentionProblem()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsAttentionProblem(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # AttentionProblem
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # AttentionProblem
    def Dtype(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def HeadDim(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def NumHeadsQ(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def NumHeadsKv(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def BatchSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def SeqlensQ(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # AttentionProblem
    def SeqlensQAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # AttentionProblem
    def SeqlensQLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttentionProblem
    def SeqlensQIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # AttentionProblem
    def SeqlensKv(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # AttentionProblem
    def SeqlensKvAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # AttentionProblem
    def SeqlensKvLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttentionProblem
    def SeqlensKvIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # AttentionProblem
    def Causal(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def Dropout(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def Alibi(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def WindowLeft(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def WindowRight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def AttnMask(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def Deterministic(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def PagedKv(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def PagedBlockSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(34))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def PagedNumBlocks(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(36))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def AppendKv(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(38))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def Api(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(40))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # AttentionProblem
    def IsTraining(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(42))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # AttentionProblem
    def SolutionFwd(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(44))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from FlashBenchData.AttentionSolution import AttentionSolution
            obj = AttentionSolution()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttentionProblem
    def SolutionBwd(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(46))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from FlashBenchData.AttentionSolution import AttentionSolution
            obj = AttentionSolution()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def AttentionProblemStart(builder):
    builder.StartObject(22)

def Start(builder):
    AttentionProblemStart(builder)

def AttentionProblemAddDtype(builder, dtype):
    builder.PrependInt8Slot(0, dtype, 0)

def AddDtype(builder, dtype):
    AttentionProblemAddDtype(builder, dtype)

def AttentionProblemAddHeadDim(builder, headDim):
    builder.PrependInt32Slot(1, headDim, 0)

def AddHeadDim(builder, headDim):
    AttentionProblemAddHeadDim(builder, headDim)

def AttentionProblemAddNumHeadsQ(builder, numHeadsQ):
    builder.PrependInt32Slot(2, numHeadsQ, 0)

def AddNumHeadsQ(builder, numHeadsQ):
    AttentionProblemAddNumHeadsQ(builder, numHeadsQ)

def AttentionProblemAddNumHeadsKv(builder, numHeadsKv):
    builder.PrependInt32Slot(3, numHeadsKv, 0)

def AddNumHeadsKv(builder, numHeadsKv):
    AttentionProblemAddNumHeadsKv(builder, numHeadsKv)

def AttentionProblemAddBatchSize(builder, batchSize):
    builder.PrependInt32Slot(4, batchSize, 0)

def AddBatchSize(builder, batchSize):
    AttentionProblemAddBatchSize(builder, batchSize)

def AttentionProblemAddSeqlensQ(builder, seqlensQ):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(seqlensQ), 0)

def AddSeqlensQ(builder, seqlensQ):
    AttentionProblemAddSeqlensQ(builder, seqlensQ)

def AttentionProblemStartSeqlensQVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSeqlensQVector(builder, numElems):
    return AttentionProblemStartSeqlensQVector(builder, numElems)

def AttentionProblemAddSeqlensKv(builder, seqlensKv):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(seqlensKv), 0)

def AddSeqlensKv(builder, seqlensKv):
    AttentionProblemAddSeqlensKv(builder, seqlensKv)

def AttentionProblemStartSeqlensKvVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSeqlensKvVector(builder, numElems):
    return AttentionProblemStartSeqlensKvVector(builder, numElems)

def AttentionProblemAddCausal(builder, causal):
    builder.PrependBoolSlot(7, causal, 0)

def AddCausal(builder, causal):
    AttentionProblemAddCausal(builder, causal)

def AttentionProblemAddDropout(builder, dropout):
    builder.PrependBoolSlot(8, dropout, 0)

def AddDropout(builder, dropout):
    AttentionProblemAddDropout(builder, dropout)

def AttentionProblemAddAlibi(builder, alibi):
    builder.PrependBoolSlot(9, alibi, 0)

def AddAlibi(builder, alibi):
    AttentionProblemAddAlibi(builder, alibi)

def AttentionProblemAddWindowLeft(builder, windowLeft):
    builder.PrependInt32Slot(10, windowLeft, 0)

def AddWindowLeft(builder, windowLeft):
    AttentionProblemAddWindowLeft(builder, windowLeft)

def AttentionProblemAddWindowRight(builder, windowRight):
    builder.PrependInt32Slot(11, windowRight, 0)

def AddWindowRight(builder, windowRight):
    AttentionProblemAddWindowRight(builder, windowRight)

def AttentionProblemAddAttnMask(builder, attnMask):
    builder.PrependBoolSlot(12, attnMask, 0)

def AddAttnMask(builder, attnMask):
    AttentionProblemAddAttnMask(builder, attnMask)

def AttentionProblemAddDeterministic(builder, deterministic):
    builder.PrependBoolSlot(13, deterministic, 0)

def AddDeterministic(builder, deterministic):
    AttentionProblemAddDeterministic(builder, deterministic)

def AttentionProblemAddPagedKv(builder, pagedKv):
    builder.PrependBoolSlot(14, pagedKv, 0)

def AddPagedKv(builder, pagedKv):
    AttentionProblemAddPagedKv(builder, pagedKv)

def AttentionProblemAddPagedBlockSize(builder, pagedBlockSize):
    builder.PrependInt32Slot(15, pagedBlockSize, 0)

def AddPagedBlockSize(builder, pagedBlockSize):
    AttentionProblemAddPagedBlockSize(builder, pagedBlockSize)

def AttentionProblemAddPagedNumBlocks(builder, pagedNumBlocks):
    builder.PrependInt32Slot(16, pagedNumBlocks, 0)

def AddPagedNumBlocks(builder, pagedNumBlocks):
    AttentionProblemAddPagedNumBlocks(builder, pagedNumBlocks)

def AttentionProblemAddAppendKv(builder, appendKv):
    builder.PrependBoolSlot(17, appendKv, 0)

def AddAppendKv(builder, appendKv):
    AttentionProblemAddAppendKv(builder, appendKv)

def AttentionProblemAddApi(builder, api):
    builder.PrependInt8Slot(18, api, 0)

def AddApi(builder, api):
    AttentionProblemAddApi(builder, api)

def AttentionProblemAddIsTraining(builder, isTraining):
    builder.PrependBoolSlot(19, isTraining, 0)

def AddIsTraining(builder, isTraining):
    AttentionProblemAddIsTraining(builder, isTraining)

def AttentionProblemAddSolutionFwd(builder, solutionFwd):
    builder.PrependUOffsetTRelativeSlot(20, flatbuffers.number_types.UOffsetTFlags.py_type(solutionFwd), 0)

def AddSolutionFwd(builder, solutionFwd):
    AttentionProblemAddSolutionFwd(builder, solutionFwd)

def AttentionProblemAddSolutionBwd(builder, solutionBwd):
    builder.PrependUOffsetTRelativeSlot(21, flatbuffers.number_types.UOffsetTFlags.py_type(solutionBwd), 0)

def AddSolutionBwd(builder, solutionBwd):
    AttentionProblemAddSolutionBwd(builder, solutionBwd)

def AttentionProblemEnd(builder):
    return builder.EndObject()

def End(builder):
    return AttentionProblemEnd(builder)
