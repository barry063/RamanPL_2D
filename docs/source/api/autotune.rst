Baseline auto-tuning
====================

The auto-tuning interface is exposed as a method on each fit class.
Pass ``method_grids`` to specify per-method parameter sweeps (v0.6.3+);
omit it to use the full 24-candidate default grid.

.. automethod:: ramanpl.mapping._raman_mapping.RamanMapping.autotune_baseline
.. automethod:: ramanpl.mapping._pl_mapping.PLMapping.autotune_baseline
.. automethod:: ramanpl.single_fit.RamanFit.RamanFit.autotune_baseline
.. automethod:: ramanpl.single_fit.PLfit.PLfit.autotune_baseline

.. automethod:: ramanpl.mapping._raman_mapping.RamanMapping.apply_choice
.. automethod:: ramanpl.mapping._pl_mapping.PLMapping.apply_choice
.. automethod:: ramanpl.single_fit.RamanFit.RamanFit.apply_choice
.. automethod:: ramanpl.single_fit.PLfit.PLfit.apply_choice

Result type
-----------

.. autoclass:: ramanpl._autotune.BaselineAutotuneResult
   :members:
   :undoc-members:
