# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FlashBenchData

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class AttentionBenchTable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AttentionBenchTable()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsAttentionBenchTable(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # AttentionBenchTable
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # AttentionBenchTable
    def Problems(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from FlashBenchData.AttentionProblem import AttentionProblem
            obj = AttentionProblem()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # AttentionBenchTable
    def ProblemsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # AttentionBenchTable
    def ProblemsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # AttentionBenchTable
    def Platform(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # AttentionBenchTable
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def AttentionBenchTableStart(builder):
    builder.StartObject(3)

def Start(builder):
    AttentionBenchTableStart(builder)

def AttentionBenchTableAddProblems(builder, problems):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(problems), 0)

def AddProblems(builder, problems):
    AttentionBenchTableAddProblems(builder, problems)

def AttentionBenchTableStartProblemsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartProblemsVector(builder, numElems):
    return AttentionBenchTableStartProblemsVector(builder, numElems)

def AttentionBenchTableAddPlatform(builder, platform):
    builder.PrependInt8Slot(1, platform, 0)

def AddPlatform(builder, platform):
    AttentionBenchTableAddPlatform(builder, platform)

def AttentionBenchTableAddVersion(builder, version):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(version), 0)

def AddVersion(builder, version):
    AttentionBenchTableAddVersion(builder, version)

def AttentionBenchTableEnd(builder):
    return builder.EndObject()

def End(builder):
    return AttentionBenchTableEnd(builder)
