"""Working demo of the ``--tt_plugin`` mechanism: ``mdata_plugin.py`` provides complete
(pack-grouped) datasets via ``provides_dataset``/``build_dataset``; ``naive_mdata_plugin.py``
supplies raw rows via ``samples``. Requires the Ant-internal ``mdatasets`` package; stays
in the internal tree."""