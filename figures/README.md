# Figure reproduction, not new measurement

Latest author-confirmed plot sources at assembly. Both plots use log time
axes in milliseconds. Styles and supplied arrays are preserved. Use
`run_experiment.ps1 -Experiment figures` to write fresh ignored local outputs.

| Source | Evidence classification | Revision |
|---|---|---|
| `mkf/MKF-multi-CPU-FPGA.py` | CPU derived from an earlier FPGA baseline/author ratios; FPGA projected/author supplied | BFV-P1 n=16: 1.29353 -> **1.290 ms**, 2026-10-28; CPU unchanged |
| `kklss/KKLSS-multi-CPU-FPGA.py` | Reference transcription and author BFV-P1 replacement | BFV-P1 FPGA `[0.530,0.814,1.493,2.380,5.129,10.482]` ms |

Adjacent CSVs retain point values and evidence status. Labels named CPU/FPGA
are preserved for visual reproduction, not measurement certification.
Do not mix figure inputs with raw experiments. Other historical resource,
rotation or network figures are not reconstructed with invented missing data.
