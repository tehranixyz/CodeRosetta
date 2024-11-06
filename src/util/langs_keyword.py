langs_keywords = {
    "cuda_keywords": [
        "__global__",
        "__device__",
        "__host__",
        "blockIdx",
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
        "blockDim",
        "blockDim.x",
        "blockDim.y",
        "blockDim.z",
        "gridDim",
        "gridDim.x",
        "gridDim.y",
        "gridDim.z",
        "__syncthreads()",
        "__shared__",
        "atomicAdd",
        "atomicSub",
        "atomicExch",
        "__ballot()",
        "__shfl()",
        "__shfl_up()",
        "__shfl_down()",
        "__shfl_xor()",
        "texture",
        "tex1Dfetch",
        "tex2D",
        "__ldg",
    ],
    "cuda_keywords_strict": [
        "threadIdx",
        "blockIdx",
        "blockDim",
    ],
    "cpp_keywords": [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ],
    "cpp_avoid": [
        "virtual",
        "goto",
        "try",
        "catch",
        "throw",
        "typeid",
        "reinterpret_cast",
        "dynamic_cast",
        "static_cast",
        "const_cast",
        "new",
        "delete",
        "namespace",
        "template",
        "friend",
        "explicit",
        "using",
        "asm",
    ],
    "fortran_keywords": [
        "ACCEPT",
        "ALLOCATABLE",
        "ALLOCATE",
        "ASSIGN",
        "ASSOCIATE",
        "BACKSPACE",
        "BLOCK",
        "BLOCKDATA",
        "BYTE",
        "CALL",
        "CASE",
        "CHARACTER",
        "CLOSE",
        "COMMON",
        "COMPLEX",
        "CONTAINS",
        "CONTINUE",
        "CYCLE",
        "DATA",
        "DEALLOCATE",
        "DEFAULT",
        "DIMENSION",
        "DO",
        "DOUBLE",
        "DOUBLEPRECISION",
        "DOWHILE",
        "ELEMENTAL",
        "ELSE",
        "ELSEIF",
        "ELSEWHERE",
        "END",
        "ENDBLOCK",
        "ENDBLOCKDATA",
        "ENDFILE",
        "ENDIF",
        "ENDINTERFACE",
        "ENDMODULE",
        "ENDPROGRAM",
        "ENDSELECT",
        "ENDSUBMODULE",
        "ENDSUBROUTINE",
        "ENDTYPE",
        "ENDWHERE",
        "ENTRY",
        "ENUM",
        "EQUIVALENCE",
        "EXIT",
        "EXTERNAL",
        "FILE",
        "FORMAT",
        "FUNCTION",
        "GENERIC",
        "GO",
        "GOTO",
        "IF",
        "IMPLICIT",
        "IMPORT",
        "IN",
        "INCLUDE",
        "INOUT",
        "INTENT",
        "INTERFACE",
        "INTRINSIC",
        "IS",
        "KIND",
        "LEN",
        "LENTRIM",
        "LOGICAL",
        "MODULE",
        "NAME",
        "NONE",
        "NULL",
        "NULLIFY",
        "ONLY",
        "OPEN",
        "OPERATOR",
        "OPTIONAL",
        "OUT",
        "PARAMETER",
        "PASS",
        "PAUSE",
        "POINTER",
        "PRECISION",
        "PRINT",
        "PRIVATE",
        "PROCEDURE",
        "PROGRAM",
        "PUBLIC",
        "PURE",
        "REAL",
        "READ",
        "RECURSIVE",
        "RESULT",
        "RETURN",
        "REWIND",
        "SAVE",
        "SELECT",
        "SEQUENCE",
        "STOP",
        "SUBMODULE",
        "SUBROUTINE",
        "TARGET",
        "THEN",
        "TO",
        "TYPE",
        "USE",
        "VALUE",
        "VOLATILE",
        "WHERE",
        "WHILE",
        "WRITE",
        "accept",
        "allocatable",
        "allocate",
        "assign",
        "associate",
        "backspace",
        "block",
        "blockdata",
        "byte",
        "call",
        "case",
        "character",
        "close",
        "common",
        "complex",
        "contains",
        "continue",
        "cycle",
        "data",
        "deallocate",
        "default",
        "dimension",
        "do",
        "double",
        "doubleprecision",
        "dowhile",
        "elemental",
        "else",
        "elseif",
        "elsewhere",
        "end",
        "endblock",
        "endblockdata",
        "endfile",
        "endif",
        "endinterface",
        "endmodule",
        "endprogram",
        "endselect",
        "endsubmodule",
        "endsubroutine",
        "endtype",
        "endwhere",
        "entry",
        "enum",
        "equivalence",
        "exit",
        "external",
        "file",
        "format",
        "function",
        "generic",
        "go",
        "goto",
        "if",
        "implicit",
        "import",
        "in",
        "include",
        "inout",
        "intent",
        "interface",
        "intrinsic",
        "is",
        "kind",
        "len",
        "lentrim",
        "logical",
        "module",
        "name",
        "none",
        "null",
        "nullify",
        "only",
        "open",
        "operator",
        "optional",
        "out",
        "parameter",
        "pass",
        "pause",
        "pointer",
        "precision",
        "print",
        "private",
        "procedure",
        "program",
        "public",
        "pure",
        "real",
        "read",
        "recursive",
        "result",
        "return",
        "rewind",
        "save",
        "select",
        "sequence",
        "stop",
        "submodule",
        "subroutine",
        "target",
        "then",
        "to",
        "type",
        "use",
        "value",
        "volatile",
        "where",
        "while",
        "write",
    ],
    # Add reserved keyword of any other languages here.
}
