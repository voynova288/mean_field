# Common order-parameter helpers

`analysis.order_parameters` contains reusable diagnostics for projected HF
states: occupations, spin/valley polarization, inter-valley coherence,
translation/fold order, and lightweight classification helpers.

System modules remain responsible for constructing labels and converting their
stored density convention to a conventional projector.  The common package does
not build Hamiltonians, run SCF, or decide physical thresholds for new systems.
Historical thresholds are preserved only in named adapters such as the TDBG
classification preset.
