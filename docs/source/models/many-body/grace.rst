GRACE
#####

Train this architecture on your own data using the :doc:`graph-pes-train <../../cli/graph-pes-train/root>` CLI, using e.g. the following config:

.. code-block:: yaml

   model:
      +GRACE:
         elements: [H, C, N, O]
         cutoff: 6.0


Definition
----------

.. autoclass:: graph_pes.models.GRACE


Replicating the reference presets
---------------------------------

The parameter names of :class:`~graph_pes.models.GRACE` follow those of the
reference `tensorpotential <https://github.com/ICAMS/grace-tensorpotential>`_
implementation. The ``small`` versions of the ``GRACE_1LAYER`` and
``GRACE_2LAYER`` presets correspond to the following configs:

.. tab-set::

    .. tab-item:: GRACE-1L (small)

        .. code-block:: yaml

            model:
                offset:
                    +LearnableOffset: {}
                many-body:
                    +GRACE:
                        elements: [H, C, N, O]
                        cutoff: 6.0
                        layers: 1
                        lmax: 3
                        n_rad_max: 20
                        max_order: 3
                        n_mlp_dens: 10

    .. tab-item:: GRACE-2L (small)

        .. code-block:: yaml

            model:
                offset:
                    +LearnableOffset: {}
                many-body:
                    +GRACE:
                        elements: [H, C, N, O]
                        cutoff: 6.0
                        layers: 2
                        lmax: [3, 2]
                        n_rad_max: [20, 32]
                        max_order: 3
                        n_mlp_dens: 8

    .. tab-item:: GRACE-1L (latest)

        .. code-block:: yaml

            model:
                offset:
                    +LearnableOffset: {}
                many-body:
                    +GRACE:
                        elements: [H, C, N, O]
                        cutoff: 6.0
                        layers: 1
                        lmax: 4
                        n_rad_max: 32
                        prod_func_n_max: 32
                        max_order: 4
                        n_mlp_dens: 10
                        envelope_order: 16
                        radial_activation: SiLU

The following differences to the reference implementation are worth noting:

* weights are not interchangeable: the real spherical harmonics and
  Clebsch-Gordan coefficients here are those of :mod:`e3nn`, and so pre-trained
  GRACE foundation models cannot (yet) be loaded into this implementation;
* linearly dependent coupling paths are not pruned from the product basis.
  This leads to a small number of redundant (but harmless) weights;
* weights are initialised so that features have (approximately) unit variance,
  following the conventions used elsewhere in ``graph-pes`` (and :mod:`e3nn`);
* per-element energy offsets are handled by an additional
  :class:`~graph_pes.models.LearnableOffset` (or
  :class:`~graph_pes.models.FixedOffset`) component, as above, rather than
  within the model itself. ZBL repulsion can similarly be added using
  :class:`~graph_pes.models.ZBLCoreRepulsion`.


Building blocks
---------------

.. autoclass:: graph_pes.models.e3nn.grace.ScalarSingleParticleBasis
.. autoclass:: graph_pes.models.e3nn.grace.ChannelwiseProduct
.. autoclass:: graph_pes.models.e3nn.grace.ChannelMixer
.. autoclass:: graph_pes.models.e3nn.grace.FunctionReduce
.. autoclass:: graph_pes.models.e3nn.grace.ProductBasis
.. autoclass:: graph_pes.models.e3nn.grace.EquivariantSingleParticleBasis
