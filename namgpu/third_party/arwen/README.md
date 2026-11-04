# ArWen files copied into woof-nam

Copied 2026-11-04 from the ArWen source tree (Python package "gpuwm", 2.9.11 line),
Apache-2.0 (LICENSE here). Copyright 2026 ArWen contributors. Unmodified copies; the woof-nam
units of mode "copy-arwen" adapt them in namgpu/physics/{myj,jsfc,noah}/ and keep this notice.

| file | ArWen path |
|---|---|
| core/myjpbl.py, core/kernels/myjpbl.cu | gpuwm/core/myjpbl.py, gpuwm/core/kernels/myjpbl.cu (MYJ PBL, WRF v4.6.1 lineage) |
| core/myjsfc.py, core/myjsfc_tables.py, core/kernels/myjsfc.cu | Eta/MYJ surface layer |
| core/noah.py, core/kernels/noah.cu, core/kernels/noah_init.cu | Noah LSM |
| verify/myj_ref.py | float32 CPU transcription of WRF MYJ/Eta (reading aid only) |

Known differences to close (FEASIBILITY.md): the NAM MYJ copy is ~26% line-identical to WRF
v4.7.1, the NAM JSFC ~80%, and NAM Noah carries the NCEP v4 changes (frozen-soil
transpiration, reduced roughness for five vegetation types) that WRF's Noah lacks. ArWen
kernels are bottom-up columns; NMMB is top-down (l=1 top).
