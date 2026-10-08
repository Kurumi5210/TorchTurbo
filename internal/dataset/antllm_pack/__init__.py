"""torchturbo plugin for antllm pre-packed SFT data.

``antllm_pack_plugin`` (``antllm_prepack``) is the ``--tt_plugin`` entry point: it
imports antllm and runs ``SFTPrePackProcesserV2`` into ``save_path`` every run
(no cache; multi-node: each node's ``LOCAL_RANK==0`` packs its share),
then delegates reading to the sibling reader module ``antllm_prepacked_reader``
(``AntllmPrePackedDataset`` + ``collate_prepacked``, imported into the pack plugin and
served via its ``DatasetPlugin.get_collate_fn``). The reader memmaps the pack directly (no
antllm at read time), recovers per-dataset sampling ratios from the ``data_path``
recorded in ``meta.json``, and emits the engine's packed-input contract.
"""
