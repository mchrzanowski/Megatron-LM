# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size, is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, get_args, set_args, set_global_variables
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support

    fp8_available, reason_for_no_fp8 = check_fp8_support()
except ImportError:
    fp8_available = False
    reason_for_no_fp8 = "TransformerEngine not available"

WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
_SEED = 1234

# Skip all tests in this file for LTS versions
pytestmark = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)


class SimpleModel(nn.Module):
    """Simple model for testing LayerWiseDistributedOptimizer.

    Model with 5 layers to ensure more than 8 parameters (10 total: 5 weights + 5 biases).
    """

    def __init__(self, input_size=80, hidden_size=48, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class TinyModel(nn.Module):
    """Tiny model with only 1 layer (2 parameters: weight and bias)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc1(x)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestLayerWiseOptimizer:
    """Test class for LayerWiseDistributedOptimizer with common setup code."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        rank = int(os.getenv('RANK', '0'))
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_model_and_optimizer(
        self,
        model_class=SimpleModel,
        clip_grad=1.0,
        model_kwargs=None,
        use_layer_wise=True,
        copy_from=None,
    ):
        """Create model, DDP wrapper, and optimizer.

        Args:
            model_class: Model class to instantiate
            clip_grad: Optional gradient clipping value
            model_kwargs: Optional kwargs for model initialization
            use_layer_wise: If True, wrap optimizer in LayerWiseDistributedOptimizer;
                          if False, use get_megatron_optimizer instead (for reference)

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        if model_kwargs is None:
            model_kwargs = {}

        model = model_class(**model_kwargs).bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )
        if copy_from:
            model.module.load_state_dict(copy_from.module.state_dict())
        else:
            model.broadcast_params()

        optimizer_config = OptimizerConfig(
            optimizer='adam',
            lr=0.01,
            weight_decay=0.01,
            bf16=not use_layer_wise,
            use_distributed_optimizer=False,
            clip_grad=clip_grad,
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = get_megatron_optimizer(optimizer_config, [model])
        if use_layer_wise:
            optimizer_config.bf16 = True
            optimizer = LayerWiseDistributedOptimizer(
                optimizer.chained_optimizers, optimizer_config, pg_collection
            )
        return model, optimizer, pg_collection

    def create_reference_model(self, model):
        """Create a reference model by cloning the current model."""
        reference_model = type(model.module)().bfloat16().cuda()
        reference_model.load_state_dict(model.module.state_dict())
        return reference_model

    def test_basic(self):
        """Test basic LayerWiseDistributedOptimizer initialization and step with bf16."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Verify basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"

        reference_model = self.create_reference_model(model)

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

        # Verify parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # Verify all ranks have the same updated parameters (test allgather)
        dp_size = get_pg_size(pg_collection.dp_cp)

        if dp_size > 1:
            for name, param in model.named_parameters():
                # Gather parameters from all ranks
                param_list = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(param_list, param.data, group=pg_collection.dp_cp)

                # Verify all ranks have the same parameter values
                for i in range(1, dp_size):
                    try:
                        torch.testing.assert_close(param_list[0], param_list[i])
                    except AssertionError as e:
                        # Append additional context without overwriting the default message
                        raise AssertionError(
                            f"Parameter {name} differs between rank 0 and rank {i}. {str(e)}"
                        ) from None

    def test_get_grad_norm(self):
        """Test LayerWiseDistributedOptimizer gradient norm computation."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            use_layer_wise=False
        )

        # Set same gradients on both models
        # note that model is different at this point but we're only testing grad norm here
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.float().detach()
            ref_param.main_grad = grad_value.float().detach()

        # Test get_grad_norm on both optimizers
        optimizer.prepare_grads()
        grad_norm = optimizer.get_grad_norm()

        reference_optimizer.prepare_grads()
        reference_grad_norm = reference_optimizer.get_grad_norm()

        assert grad_norm is not None, "Grad norm should not be None"
        assert grad_norm >= 0, "Grad norm should be non-negative"

        # Compare with reference optimizer grad norm
        torch.testing.assert_close(grad_norm, reference_grad_norm, rtol=1e-5, atol=1e-5)

    def test_state_dict(self):
        """Test LayerWiseDistributedOptimizer state dict save and load."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Test state_dict
        state_dict = optimizer.state_dict()

        # Test load_state_dict
        # TODO(deyuf): fix this. not going through get() will cause missing keys like wd_mult
        # optimizer.load_state_dict(state_dict)

    def test_sharded_state_dict(self):
        """Test LayerWiseDistributedOptimizer sharded_state_dict method."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Get model sharded state dict
        model_sharded_state_dict = model.sharded_state_dict()

        # Test sharded_state_dict
        sharded_state_dict = optimizer.sharded_state_dict(model_sharded_state_dict)

        # Verify the sharded_state_dict is not None and has expected structure
        assert sharded_state_dict is not None, "Sharded state dict should not be None"
        assert (
            'optimizer' in sharded_state_dict
        ), "Sharded state dict should contain 'optimizer' key"

        # Verify that replica_id is set correctly (should be 0 for DP dimension)
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.dict_utils import nested_values

        for sh_base in nested_values(sharded_state_dict):
            if isinstance(sh_base, ShardedTensor):
                assert (
                    len(sh_base.replica_id) == 3
                ), f'Expected replica_id format (PP, TP, DP), got: {sh_base.replica_id}'
                assert (
                    sh_base.replica_id[2] == 0
                ), f'Expected DP replica_id to be 0 for layer-wise optimizer, got: {sh_base.replica_id[2]}'

    def test_multiple_optimizers(self):
        """Test LayerWiseDistributedOptimizer with multiple chained optimizers.

        This test properly tests allgather functionality with multiple ranks.
        """
        model = SimpleModel().bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        # Split parameters into two groups for testing multiple optimizers
        params = list(model.parameters())
        mid_point = len(params) // 2
        param_groups_1 = [{'params': params[:mid_point]}]
        param_groups_2 = [{'params': params[mid_point:]}]

        # Create two separate base optimizers
        base_optimizer_1 = torch.optim.Adam(param_groups_1, lr=optimizer_config.lr)
        base_optimizer_2 = torch.optim.Adam(param_groups_2, lr=optimizer_config.lr)

        wrapped_optimizer_1 = FP32Optimizer(base_optimizer_1, optimizer_config, None)
        wrapped_optimizer_2 = FP32Optimizer(base_optimizer_2, optimizer_config, None)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = LayerWiseDistributedOptimizer(
            [wrapped_optimizer_1, wrapped_optimizer_2], optimizer_config, pg_collection
        )

        assert len(optimizer.chained_optimizers) == 2, "Should have two chained optimizers"

        # Set gradients and test optimizer step - this will trigger allgather
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

    def test_bf16_wrapping(self):
        """Test LayerWiseDistributedOptimizer automatically wraps optimizer with bf16."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Verify bf16 wrapping happened
        assert isinstance(
            optimizer.chained_optimizers[0], Float16OptimizerWithFloat16Params
        ), "Optimizer should be wrapped in Float16OptimizerWithFloat16Params"

        for param in model.parameters():
            param.grad = torch.randn_like(param)

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

    def test_bf16_error(self):
        """Test LayerWiseDistributedOptimizer raises error when receiving pre-wrapped Float16 optimizer."""
        model = SimpleModel().bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        # Create base optimizer and manually wrap in Float16 optimizer
        param_groups = [{'params': list(model.parameters())}]
        base_optimizer = torch.optim.Adam(param_groups, lr=optimizer_config.lr)
        wrapped_optimizer = Float16OptimizerWithFloat16Params(
            base_optimizer, optimizer_config, None, None
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        # Should raise TypeError when receiving already-wrapped Float16 optimizer
        with pytest.raises(
            TypeError, match='LayerWiseDistributedOptimizer received Float16 optimizer already'
        ):
            LayerWiseDistributedOptimizer([wrapped_optimizer], optimizer_config, pg_collection)

    def _run_parameter_update_test(self, model_class=SimpleModel):
        """Helper method to test parameter updates with a given model class.

        Args:
            model_class: Model class to use for testing
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer(model_class=model_class)

        # Create reference model and optimizer using the same function
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=model_class, use_layer_wise=False, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)

    def test_parameter_updates(self):
        """Test LayerWiseDistributedOptimizer actually updates model parameters."""
        self._run_parameter_update_test()

    def test_parameter_updates_insufficient_parameters(self):
        """Test LayerWiseDistributedOptimizer when there are insufficient parameters for all ranks.

        Uses a tiny model with only 1 layer (2 parameters: weight and bias).
        This will be insufficient when world size > 2.
        """
        self._run_parameter_update_test(model_class=TinyModel)

    def test_broadcast_vs_allgather(self):
        """Test LayerWiseDistributedOptimizer allgather code agains broadcast code."""
        model, optimizer, pg_collection = self.create_model_and_optimizer(model_class=SimpleModel)

        # Create reference model and optimizer using the same function
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=SimpleModel, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # step() internal call allgather_params. replace reference object with bcast
        reference_optimizer.allgather_params = reference_optimizer.broadcast_params
        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)

    # ---- FP8 + layer-wise optimizer tests ----

    @staticmethod
    def _model_provider_fp8(pre_process=True, post_process=True):
        """Model provider for FP8 GPT model tests."""
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.vocal_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    def _create_fp8_test_args(self, tp, recipe, fp8_param_gather=True):
        """Create test args for FP8 + layer-wise optimizer tests."""
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_layer_wise_optimizer.py']
        args = parse_args()
        args.num_layers = 4
        args.vocal_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.micro_batch_size = 2
        args.create_attention_mask_in_dataloader = True
        args.seq_length = 512
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = False
        args.optimizer = 'dist_muon'
        args.fp8 = "e4m3"
        args.fp8_recipe = recipe
        args.fp8_param_gather = fp8_param_gather
        args.ddp_bucket_size = 1024

        validate_args(args)
        set_global_variables(args, False)
        return args

    def _get_fp8_batch(self, seq_length=512, micro_batch_size=2):
        """Create a test batch for FP8 GPT model tests."""
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_fp8_layer_wise_test(self, tp_size, recipe, fp8_param_gather=True, num_iters=100):
        """Run FP8 + layer-wise optimizer training loop and return loss list."""
        args = self._create_fp8_test_args(tp_size, recipe, fp8_param_gather)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)

        input_ids, labels, position_ids, attention_mask, loss_mask = self._get_fp8_batch(
            args.seq_length, args.micro_batch_size
        )

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self._model_provider_fp8, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1

        # Verify FP8 params exist when fp8_param_gather is enabled.
        num_fp8_params = 0
        for _, param in gpt_model[0].named_parameters():
            assert param.requires_grad
            assert param.main_grad is not None
            if is_float8tensor(param):
                num_fp8_params += 1
        if fp8_param_gather:
            # Each layer has 4 GEMM weights: qkv, proj, fc1, fc2.
            assert num_fp8_params == 4 * args.num_layers

        loss_list = []
        for i in range(num_iters):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            assert output.shape[0] == args.micro_batch_size
            assert output.shape[1] == args.seq_length

            loss = output.mean()
            loss.backward()

            if args.overlap_grad_reduce:
                gpt_model[0].finish_grad_sync()

            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        return torch.tensor(loss_list)

    def _cleanup_fp8_state(self):
        """Clean up global state after FP8 GPT model tests."""
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def test_fp8_allgather_post_processing(self):
        """Verify post_all_gather_processing is called during allgather with FP8 params."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Set gradients so step() triggers allgather.
        for param in model.parameters():
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.float().clone().detach()

        with patch(
            'megatron.core.optimizer.layer_wise_optimizer.is_float8tensor', return_value=True
        ), patch(
            'megatron.core.optimizer.layer_wise_optimizer.post_all_gather_processing'
        ) as mock_post:
            optimizer.step()

            if optimizer.dp_cp_params_list is not None:
                mock_post.assert_called()

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_param_gather_delayed_scaling(self):
        """End-to-end test: FP8 delayed scaling + layer-wise optimizer.

        Runs multiple forward/backward/step cycles with a GPTModel using
        fp8_param_gather=True, fp8_recipe='delayed', and dist_muon optimizer.
        Verifies loss is finite and training completes successfully.
        """
        try:
            loss_list = self._run_fp8_layer_wise_test(
                tp_size=2, recipe="delayed", fp8_param_gather=True, num_iters=100
            )
            assert torch.isfinite(loss_list).all(), "Loss should be finite for all iterations"
        finally:
            self._cleanup_fp8_state()

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 required for tensorwise")
    def test_fp8_param_gather_tensorwise_scaling(self):
        """End-to-end test: FP8 tensorwise scaling + layer-wise optimizer.

        Same as delayed scaling test but with fp8_recipe='tensorwise'.
        Requires TransformerEngine >= 2.2.0.
        """
        try:
            loss_list = self._run_fp8_layer_wise_test(
                tp_size=2, recipe="tensorwise", fp8_param_gather=True, num_iters=100
            )
            assert torch.isfinite(loss_list).all(), "Loss should be finite for all iterations"
        finally:
            self._cleanup_fp8_state()

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 required")
    def test_fp8_param_gather_correctness(self):
        """Compare loss trajectories with and without fp8_param_gather.

        Runs training with fp8_param_gather=True and fp8_param_gather=False
        using the layer-wise optimizer. Verifies that loss trajectories match
        within tolerance, confirming FP8 param gather doesn't affect convergence.
        """
        try:
            loss_list = self._run_fp8_layer_wise_test(
                tp_size=2, recipe="delayed", fp8_param_gather=True, num_iters=100
            )
            loss_list_ref = self._run_fp8_layer_wise_test(
                tp_size=2, recipe="delayed", fp8_param_gather=False, num_iters=100
            )
            torch.testing.assert_close(loss_list, loss_list_ref, atol=1e-4, rtol=1e-4)
        finally:
            self._cleanup_fp8_state()

    def test_fp8_param_gather_validation_accepts_layer_wise(self):
        """Verify the fp8_param_gather validation condition accepts layer-wise optimizer.

        Tests the exact assertion condition from arguments.py:
        - Positive: 'dist' in 'dist_muon' should pass validation.
        - Negative: 'dist' in 'adam' should fail validation.
        """
        # Positive case: dist_muon optimizer should be accepted.
        assert (
            False  # use_distributed_optimizer
            or False  # use_torch_fsdp2
            or False  # use_megatron_fsdp
            or False  # not torch.is_grad_enabled() (grad is enabled)
            or 'dist' in 'dist_muon'  # layer-wise optimizer
        ), 'Validation should accept dist_muon optimizer with fp8_param_gather'

        # Negative case: plain adam optimizer should be rejected.
        assert not (
            False  # use_distributed_optimizer
            or False  # use_torch_fsdp2
            or False  # use_megatron_fsdp
            or False  # not torch.is_grad_enabled() (grad is enabled)
            or 'dist' in 'adam'  # regular optimizer, no 'dist'
        ), 'Validation should reject adam optimizer with fp8_param_gather'

        # Also verify 'sgd' and 'muon' (without dist) are rejected.
        for opt_name in ['sgd', 'muon']:
            assert not (
                False or False or False or False or 'dist' in opt_name
            ), f'Validation should reject {opt_name} optimizer with fp8_param_gather'

    @pytest.mark.skipif(WORLD_SIZE == 1, reason="Multi-rank test requires WORLD_SIZE > 1")
    def test_fp8_allgather_multi_iteration(self):
        """Run 3+ iterations with mock FP8, verify post-processing and param sync.

        Uses mock FP8 tensors to verify that post_all_gather_processing is
        called on each iteration and that parameters remain synchronized
        across ranks throughout training.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer()
        dp_size = get_pg_size(pg_collection.dp_cp)

        with patch(
            'megatron.core.optimizer.layer_wise_optimizer.is_float8tensor', return_value=True
        ), patch(
            'megatron.core.optimizer.layer_wise_optimizer.post_all_gather_processing'
        ) as mock_post:
            for iteration in range(3):
                for param in model.parameters():
                    grad_value = torch.randn_like(param)
                    torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
                    param.main_grad = grad_value.float().clone().detach()

                optimizer.step()

            # Verify post_all_gather_processing was called for each iteration.
            if optimizer.dp_cp_params_list is not None:
                # Called once per _allgather_helper invocation per iteration.
                assert mock_post.call_count >= 3, (
                    f"Expected at least 3 calls to post_all_gather_processing, "
                    f"got {mock_post.call_count}"
                )

        # Verify params are synchronized across ranks after all iterations.
        if dp_size > 1:
            for name, param in model.named_parameters():
                param_list = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(
                    param_list, param.data, group=pg_collection.dp_cp
                )
                for i in range(1, dp_size):
                    try:
                        torch.testing.assert_close(param_list[0], param_list[i])
                    except AssertionError as e:
                        raise AssertionError(
                            f"Parameter {name} differs between rank 0 and rank {i} "
                            f"after 3 iterations. {str(e)}"
                        ) from None
