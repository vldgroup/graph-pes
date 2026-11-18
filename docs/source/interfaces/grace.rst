GRACE foundation models
=======================

``graph-pes`` can load the two-layer
`GRACE foundation models <https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/>`__
(e.g. ``GRACE-2L-SMAX-OMAT-medium`` and ``GRACE-2L-OMAT-large-ft-E``) and use
them like any other :class:`~graph_pes.GraphPESModel`, either via the Python API:

.. code-block:: python

   from graph_pes.interfaces import grace_fm
   model = grace_fm("GRACE-2L-SMAX-OMAT-medium")
   calc = model.ase_calculator()

... or within a ``graph-pes-train`` configuration file, e.g. to fine-tune:

.. code-block:: yaml

   model:
      offset: +LearnableOffset()
      many-body:
         +grace_fm:
            model: GRACE-2L-SMAX-OMAT-medium

   fitting:
      auto_fit_reference_energies: true
      optimizer:
         name: AdamW
         lr: 0.0001

No TensorFlow or ``tensorpotential`` installation is needed: the model's
``kokkos.npz`` export (about 240 MB) is downloaded from
`Hugging Face <https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models>`__
on first use, cached in ``~/.cache/graph-pes/grace`` (override with the
``GRAPH_PES_CACHE`` environment variable), and re-built in pure PyTorch. You can
also pass the path to a file you have already downloaded:

.. code-block:: python

   model = grace_fm(
       "GRACE-2L-OMAT-large-ft-E",
       path="path/to/GRACE-2L-OMAT-large-ft-E-kokkos.npz",
   )

In double precision, the re-implemented models reproduce the original
``tensorpotential`` (v0.6.2.1) predictions to within 1e-7 eV/atom for energies
and 1e-5 eV/Å for forces, across bulk, alloy, surface, molecular and dimer
structures. The resulting models support TorchScript, and hence LAMMPS
deployment via ``graph-pes``. They are created in ``torch.get_default_dtype()``; call
``model.to(torch.float64)`` for double precision.

.. note::

   The GRACE foundation models are released by the ICAMS group under the
   Academic Software License. By downloading them you agree to its terms.

Notes
-----

* Supported models are listed in
  ``graph_pes.interfaces._grace.GRACE_2L_FOUNDATION_MODELS``: all two-layer
  SMAX, SMAX-OMAT and OMAT medium / large models. One- and three-layer models
  are not yet supported.
* The SMAX models use element-pair-specific cutoffs between 5.0 and 7.5 Å: the
  ``graph-pes`` cutoff is the largest of these, and each bond is then cut off at
  its own radius.
* The CALM uncertainty (γ) of the reference implementation is not available.

If you use these models, please cite the GRACE papers listed in the
`gracemaker documentation <https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/>`__,
including:

.. code-block:: bibtex

    @article{Bochkarev2024GRACE,
        title = {Graph Atomic Cluster Expansion for Semilocal Interactions
                 beyond Equivariant Message Passing},
        author = {Bochkarev, Anton and Lysogorskiy, Yury and Drautz, Ralf},
        journal = {Physical Review X},
        volume = {14},
        pages = {021036},
        year = {2024},
        doi = {10.1103/PhysRevX.14.021036},
    }

Interface
---------

.. autofunction:: graph_pes.interfaces.grace_fm
