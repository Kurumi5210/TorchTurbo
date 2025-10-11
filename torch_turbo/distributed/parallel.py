from datetime import timedelta
from typing import Callable, List, Optional

import torch


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None
# Expert tensor parallel group that current rank belongs to.
_EXPERT_TENSOR_PARALLEL_GROUP = None
# Expert tensor and model combined parallel group
_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = None
# A list of global ranks for each tensor model parallel group to ease calculation of
# the first local rank in the tensor model parallel group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        if 'net_name' in nccl_comm_cfgs[pg_name]:
            nccl_options.config.net_name = nccl_comm_cfgs[pg_name].get('net_name')
            # verify net_name value
            if nccl_options.config.net_name.lower() not in ['ib', 'socket']:
                raise RuntimeError(
                    f"net_name ({nccl_options.config.net_name}) is not supported."
                    f"Accepted values: 'IB' or 'socket'."
                )
        return nccl_options
    else:
        return None

def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
    """Creates a ProcessGroup."""
    kwargs = {
        'ranks': ranks,
        'timeout': timeout,
        'backend': backend,
        'pg_options': pg_options,
        'use_local_synchronization': use_local_synchronization,
        'group_desc': group_desc,
    }
    return torch.distributed.new_group(**kwargs)

def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example,
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).

        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):
        """
        This function solve the math problem below:
            There is an equation:
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will be used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks

class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        assert (
            ep == 1 or cp == 1
        ), "Both EP and CP > 1 in not allow in one rank generator. \
            CP is only included in default RankGenerator, and EP only in expert RankGenerator."

        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order
        order = order.lower()

        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't"
                    f"specified the order ({self.order})."
                )
            elif name not in order:
                order = order + '-' + name

        self.order = order
        self.ordered_size = []

        for token in order.split('-'):
            self.ordered_size.append(self.name_to_size[token])

    def get_mask(self, order: str, token: str):
        """Create a mask for the specified tokens based on the given order.

        Args:
            order (str): The order of parallelism types (e.g., 'tp-dp-pp').
            token (str): The specific parallelism types to include in the mask,
                         separated by hyphens (e.g., 'tp-dp').
        """
        ordered_token = order.split('-')
        token_list = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token_list:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token):
        """Get rank group by input token.

        Args:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, self.ordered_size, mask)
        if self.rank_offset > 0:
            for rank_group in ranks:
                for i in range(len(rank_group)):
                    rank_group[i] += self.rank_offset
        return ranks

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: int = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    create_gloo_process_groups: bool = True,
) -> None:
    # pylint: disable=line-too-long
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            DEPRECATED. For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        pipeline_model_parallel_comm_backend (str, optional):
            The backend to use for pipeline parallel communication.
            If None, the default backend will be used.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        num_distributed_optimizer_instances (int, default = 1):
            The number of distributed optimizer replicas across the data-
            parallel domain.

        expert_tensor_parallel_size (int, default = tp_size):
            The number of GPUs to split individual tensors of expert.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

        encoder_tensor_model_parallel_size (int, default = 0):
            The number of GPUs to split individual tensors across in the encoder. If 0,
            then we use the default, decoder's tensor model parallel size.

        encoder_pipeline_model_parallel_size (int, default = 0):
            The number of tensor parallel GPU groups to allocate to the encoder. As an example,
            if pipeline_model_parallel_size is 4 and encoder_pipeline_model_parallel_size is 2,
            then the encoder will use the first two pipeline stages for its layers, and the total
            amount of pipelineing is 6.

        get_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group and returns
            those ranks that should have embeddings.

        get_position_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group, and returns
            those ranks that should have position embeddings.

        create_gloo_process_groups (bool, default = True):
            Create Gloo process groups if set to True. If set to False, Gloo process groups are
            not created and calls to get Gloo process groups will result in assertion errors.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if encoder_tensor_model_parallel_size == 0 and encoder_pipeline_model_parallel_size > 0:
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    if encoder_pipeline_model_parallel_size > 0:
        global _PIPELINE_MODEL_PARALLEL_DECODER_START
        _PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    encoder_model_size = (
        encoder_tensor_model_parallel_size
        * encoder_pipeline_model_parallel_size
        * context_parallel_size
    )
    decoder_model_size = (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {total_model_size}")

    data_parallel_size: int = world_size // total_model_size

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    assert (
        encoder_world_size + decoder_world_size == world_size
    ), f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}"

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    # TODO: support expert specific ordering
    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp.

        Let's say we have a total of 6 GPUs denoted by g0 ... g5.
        For encoder_tp=1, encoder_pp=1, decoder_tp=2, decoder_pp=1, dp=2,
        g0, g1 belong to encoder and g2, ..., g5 belong to decoder.
        The present function will create with "tp-dp-pp":
        3 data-parallel groups: [g0, g1], [g2, g4], [g3, g5]
        4 tensor model-parallel groups: [g0], [g1], [g2, g3], [g4, g5]
        4 pipeline model-parallel groups: [g0, g2], [g0, g3], [g1, g4], [g1, g5]
        """
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == 'pp':
            # Map one encoder tp rank to several decoder tp ranks, because
            # encoder tp and decoder tp won't be the same size.
            # Assign this way to avoid getting the DP ranks mixed up with the PP ranks.
            # For example, if e_ranks = [0,1,2] and d_ranks = [3,4,5,6]
            # Should yield [0,3], [0,4], [1,5], [2,6]
            rep = len(d_ranks) // len(e_ranks)
            remain = len(d_ranks) % len(e_ranks)
            e_ind = 0
            e_rep = rep + int(e_ind < remain)
            for i, y in enumerate(d_ranks):
                x = e_ranks[e_ind]
                e_rep -= 1
                if e_rep == 0:
                    e_ind += 1
                    e_rep = rep + int(e_ind < remain)
                yield x + y
        elif group_type == 'tp-pp':
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    '''
    # Build the data-parallel groups. (dp, dp-cp)
    # Build the context-parallel groups. (cp)
    # Build the model-parallel groups. (tp-pp)
    # Build the tensor model-parallel groups. (tp)
    # Build the pipeline model-parallel groups and embedding groups (pp)
    # (first and last rank in each pipeline model-parallel group).
    # Build the tensor + data parallel groups. (tp-dp-cp, tp-dp, tp-cp)
    '''

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for ranks in generator_wrapper('tp'):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp', nccl_comm_cfgs),
            group_desc='TENSOR_MODEL_PARALLEL_GROUP',
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks


    ### Expert-related parallel groups initialization
    # Build the expert model parallel group
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    for ranks in generator_wrapper('ep', is_expert=True):
        group = create_group(
            ranks,
            pg_options=get_nccl_options('ep', nccl_comm_cfgs),
            group_desc='EXPERT_MODEL_PARALLEL_GROUP',
        )
        if rank in ranks:
            _EXPERT_MODEL_PARALLEL_GROUP = group

    # Build the expert tensor parallel group
    global _EXPERT_TENSOR_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_PARALLEL_GROUP is None
    ), 'Expert tensor model parallel group is already initialized'
    for ranks in generator_wrapper('tp', is_expert=True):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('ep_tp', nccl_comm_cfgs),
            group_desc='EXPERT_TENSOR_PARALLEL_GROUP',
        )
        if rank in ranks:
            _EXPERT_TENSOR_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    assert (
        _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is None
    ), 'Expert tensor + model parallel group is already initialized'
    for ranks in generator_wrapper('tp-ep', is_expert=True):
        group = create_group(
            ranks,
            timeout=timeout,
            pg_options=get_nccl_options('tp_ep_mp', nccl_comm_cfgs),
            group_desc='EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
        )
        if rank in ranks:
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group

def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_rank():
    """Return caller's rank for the tensor-model-parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_MODEL_PARALLEL_GROUP is not None
        ), 'expert model parallel group is not initialized'
    return _EXPERT_MODEL_PARALLEL_GROUP

def get_expert_model_parallel_world_size():
    """Return world size for the expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_expert_model_parallel_group())
    else:
        return 0
    
def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_model_parallel_group())
    else:
        return 0
    
def get_expert_tensor_parallel_group(check_initialized=True):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_PARALLEL_GROUP is not None
        ), 'Expert tensor parallel group is not initialized'
    return _EXPERT_TENSOR_PARALLEL_GROUP
    
def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    return torch.distributed.get_rank(group=get_expert_tensor_parallel_group())

def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    if check_initialized:
        assert (
            _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None
        ), 'Expert tensor and model parallel group is not initialized'
    return _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP

def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    return torch.distributed.get_world_size(group=get_expert_tensor_parallel_group())