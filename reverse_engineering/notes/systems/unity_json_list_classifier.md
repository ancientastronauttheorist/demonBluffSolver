# Native List classification

The pinned engine audit executes `0x75F2B0` instead of the earlier collection
classification service. Eighteen classifier fixtures and six joined List
save/read cases pass. Twenty exact instruction/export assertions cover the
classifier and the complete runtime-export initializer unwind families.

The native predicate obtains the class name, compares seven bytes against the
exact `List`1` plus NUL literal, then requires the class's image to equal the
runtime corlib image. It does not query a namespace, generic arguments, field
layout or inheritance. Both a mismatching name and a mismatching image reject
the supplied class. Case changes, another arity, longer prefixes and empty names
are explicitly tested. Authored bytes after an embedded NUL do not participate
in the class-name comparison.

`il2cpp_class_get_image` and `il2cpp_get_corlib` are independently bound to their
exact initializer literals and slots. Class name/image/corlib remain supplied
metadata; actual comparison and control flow execute. Joined fixtures exercise
null construction, existing Lists and normal writing with this native predicate.

Both return stubs write AL only. The audit validates the low return byte and
records nonzero upper bits left from pointer-valued metadata calls. Treating
the complete RAX value as a canonical Boolean would be incorrect.

Runtime metadata discovery, managed constructor execution, allocation, GC/cache
services and compound elements remain open. This predicate supplies no proof
that an arbitrary matching class has a coherent List backing layout.

```powershell
python reverse_engineering/scripts/audit_unity_json_list_classifier.py GAME_ROOT --output REPORT
```

Report: `f530404b0f3f_807de4a83df4_unity_json_list_classifier.json`. Private
Unicorn 2.1.4 dependencies are required. No native bytes or decompiled bodies are
retained; this engine audit adds no Assembly-CSharp classification.
