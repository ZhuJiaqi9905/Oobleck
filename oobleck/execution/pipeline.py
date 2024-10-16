from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed
import torch.fx
from deepspeed import comm as dist
from deepspeed.runtime.lr_schedules import WarmupLR
from deepspeed.runtime.pipe import schedule
from torch.distributed import ProcessGroup, Work
from torch.optim import AdamW
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.execution.dataloader import OobleckDataLoader, OobleckSampler
from oobleck.execution.layer import Layer
from oobleck.execution.utils import DTYPE_TO_ID, ID_TO_DTYPE, zero_grads
from oobleck.module.model import OobleckModel


class OobleckPipelineSchedule(schedule.TrainSchedule):
    """A schedule for training a batch using pipeline parallelism.

    Unlike existing :class:`deepspeed.runtime.pipe.schedule.TrainSchedule`,
    :class:`OobleckPipelineSchedule` decouples allreduce synchronization and optimizer step
    from pipeline execution and only schedules computation part and intermediate p2p operations.

    reducing (tied) gradients and optimizer step must be done separately.
    """

    def steps(self):
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            cmds = []

            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                    self.prev_stage
                ):
                    cmds.append(schedule.SendGrad(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                    self.prev_stage
                ):
                    cmds.append(schedule.RecvActivation(curr_buffer))

            else:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(
                    self.next_stage
                ):
                    cmds.append(schedule.RecvGrad(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(
                    self.next_stage
                ):
                    cmds.append(schedule.SendActivation(prev_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(schedule.LoadMicroBatch(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(schedule.ForwardPass(curr_buffer))
                else:
                    cmds.append(schedule.BackwardPass(curr_buffer))

            # No reduce and optimizer step here at the end of the batch

            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds


class PipelineExecution:
    """
    Pipeline execution module that this rank will use for training.
    For a single stage where this rank is in, there might be several ranks in FSDP group.

    TODO: explain shard_id. Heterogeneous pipeline could have different number of GPUs for the same layer.
    """

    def __init__(
        self,
        pipeline: OobleckPipeline,
        layers: list[Layer],
        shard_id: int,
        dataloader: OobleckDataLoader,
        training_args: TrainingArguments,
    ):
        self._pipeline = weakref.ref(pipeline)
        self._layers = layers
        self._shard_id = shard_id
        self._dataloader = dataloader
        self._data_iterator = iter(self._dataloader)
        self._training_args = training_args

        # stores the loss for the current microbatch being processed
        self._loss: torch.Tensor | None = None

        # stores the loss for the entire batch
        self.total_loss: torch.Tensor | None = None

        # TODO: use HF arguments to initialize optimizer and LR properly
        self._optimizer = AdamW(
            [l._param_handle.flat_param for l in layers],
            lr=self._training_args.learning_rate,
            betas=(self._training_args.adam_beta1, self._training_args.adam_beta2),
            eps=self._training_args.adam_epsilon,
            fused=True,
        )
        num_training_steps = len(self._dataloader)
        self._lr_scheduler = WarmupLR(
            self._optimizer, self._training_args.get_warmup_steps(num_training_steps)
        )

    @property
    def pipeline(self) -> OobleckPipeline:
        return self._pipeline()

    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L2454
    def _prepare_input(self, data: torch.Tensor | Any) -> torch.Tensor | Any:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            # print(f"inputs type: {data.dtype}")
            data = data.clone().detach().to(self.pipeline.device)
            data.requires_grad = data.is_floating_point()
            return data
        return data

    # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L2472
    def _prepare_inputs(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[torch.Tensor | Any]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        return tuple(self._prepare_input(t) for _, t in inputs.items())

    def load_microbatch(self, buffer_id: int):
        assert (
            self.pipeline.is_first_stage() or self.pipeline.is_last_stage()
        ), "load_microatch can only be called at either the first stage or the last stage."

        if self.pipeline.is_first_stage():
            batch = next(self._data_iterator)
            self.pipeline.pipe_buffers["inputs"][buffer_id] = self._prepare_inputs(
                batch
            )

    def forward_pass(self, buffer_id: int):
        ori_inputs: tuple[torch.Tensor, ...] = self.pipeline.pipe_buffers["inputs"][
            buffer_id
        ]
        zero_grads(ori_inputs)
        # print(f"buffer id: {buffer_id}. is_last_stage: {self.pipeline.is_last_stage()}")
        # XXX Hack
        # Some tensor might be converted from torch.Size().
        # Convert it to torch.Size so that forward can be executed
        # inputs: tuple[torch.Size | torch.Tensor] = tuple(
        #     [
        #         torch.Size(input.tolist())
        #         if input.dim() == 1
        #         and input.data[0] == self._training_args.per_device_train_batch_size
        #         else input
        #         for input in ori_inputs
        #     ]
        # )
        inputs: list[torch.Size | torch.Tensor] = []
        i = 0
        for input in ori_inputs:
            # if input.dim() == 1:
                # print(f"data {input.data[0]}, batch_size {self._training_args.per_device_train_batch_size}. id {i}")
            
            # hack "-1"  
            if input.dim() == 1 and (input.data[0] == self._training_args.per_device_train_batch_size or input.data[0] == -1):
                inputs.append(torch.Size(input.tolist()))
                # print(f"tuple input {input}")
            else:
                inputs.append(input)
            i += 1
        inputs: tuple[torch.Size | torch.Tensor] = tuple(inputs)

        # print(f"converted inputs: {inputs}")
        # Execute forward
        for layer in self._layers:
            inputs = layer(inputs)
        outputs = inputs

        # Optionally compute loss on the last stage
        if self.pipeline.is_last_stage():
            self._loss = outputs[0]

            assert isinstance(self._loss, torch.Tensor)
            if self.total_loss is None:
                self.total_loss = torch.zeros_like(self._loss)
            self.total_loss += self._loss.detach()

        else:
            # XXX Hack
            # It might includes torch.Size() in outputs.
            # Convert it to torch.Tensor so that it can be transferred
            # outputs: tuple[torch.Tensor] = tuple(
            #     [
            #         output
            #         if torch.is_tensor(output)
            #         else torch.LongTensor(data=output).to(self.pipeline.device)
            #         for output in outputs
            #     ]
            # )
            trans_outputs: list[torch.Tensor] = []
            i = 0
            for output in outputs:
                # print(f"type of output: {type(output)}") 
                if torch.is_tensor(output):
                    trans_outputs.append(output)
                else:
                    # 这里把一个非tensor类型转换为tensor了
                    # print(f"output is not tensor: {output}. id = {i}, type {type(output)}.")
                    trans_outputs.append(torch.LongTensor(data=output).to(self.pipeline.device))
                i += 1
            outputs = tuple(trans_outputs)


            self.pipeline.pipe_buffers["outputs"][buffer_id] = outputs

    def backward_pass(self, buffer_id: int):
        if self.pipeline.is_last_stage():
            loss = self._loss
            self._layers[-1].backward(loss)
        else:
            output_tensors: tuple[torch.Tensor] = self.pipeline.pipe_buffers["outputs"][
                buffer_id
            ]
            output_tensors = tuple([t for t in output_tensors if t.requires_grad])
            grad_tensors: tuple[
                torch.Tensor
            ] = self.pipeline.communication.grad_recv_buf

            # Oobleck sharded model always returns tuple with tensors and torch.Size.
            assert len(output_tensors) == len(grad_tensors)

            self._layers[-1].backward((output_tensors, grad_tensors))

        # Free up memory from the output of forward()
        self.pipeline.pipe_buffers["outputs"][buffer_id] = None
        grad_tensors = None
        self._loss = None

    def optimizer_step(self, lr_kwargs=None):
        # amp enable check: gradient clipping
        # print("before optimizer step")
        self._optimizer.step()
        # print("after optimizer step")
        self._lr_scheduler.step(**(lr_kwargs or {}))
        # print("after lr_scheduler step")


class PipelineCommunication:
    def __init__(
        self,
        pipeline: OobleckPipeline,
        process_group: ProcessGroup,
        prev_rank: int | None,
        next_rank: int | None,
    ):
        self._pipeline = weakref.ref(pipeline)
        # 一个模型因为pipeline parallel被分到多个node上，这些node组成的process_group
        self._process_group = process_group
        self.prev_rank = prev_rank
        self.next_rank = next_rank

        self.sent_activation_meta: bool = False
        # initialized in :func:`oobleck.execution.PipelineCommunication.recv_activations`.
        self.activation_recv_buf: tuple[torch.Tensor] | None = None
        # initialized in :func:`oobleck.execution.PipelineCommunication.recv_gradients`.
        self.grad_recv_buf: tuple[torch.Tensor] | None = None

    @property
    def pipeline(self) -> OobleckPipeline:
        return self._pipeline()

    def _send(
        self, tensor: torch.Tensor, dest_rank: int, async_op: bool = False
    ) -> Work:
        return (
            dist.isend(tensor, dest_rank, self._process_group)
            if async_op
            else dist.send(tensor, dest_rank, self._process_group)
        )

    def _recv(
        self, tensor: torch.Tensor, src_rank: int, async_op: bool = False
    ) -> Work:
        return (
            dist.irecv(tensor, src_rank, self._process_group)
            if async_op
            else dist.recv(tensor, src_rank, self._process_group)
        )

    def send_activations(self, buffer_id: int):
        def _send_activation_meta(buffer: tuple[torch.Tensor], receiver_rank: int):
            """Send activation dimension first to the next stage
            so that it can initialize buffers.

            Metadata is communicated in this order:
                * num_tensors in tensor tuple
                foreeach tensor in buffer:
                    * ndims
                    * dtype
                    * shape
                    * requires_grad
            """
            assert isinstance(
                buffer, tuple
            ), f"Could not send meta type {type(buffer)}."
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.pipeline.device)
            self._send(count_tensor, receiver_rank)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(
                    self.pipeline.device
                )
                send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(
                    self.pipeline.device
                )
                send_shape = torch.LongTensor(data=tensor.size()).to(
                    self.pipeline.device
                )
                send_req_grad = torch.LongTensor(
                    data=[1 if tensor.requires_grad else 0]
                ).to(self.pipeline.device)
                self._send(send_ndims, receiver_rank)
                self._send(send_dtype, receiver_rank)
                self._send(send_shape, receiver_rank)
                self._send(send_req_grad, receiver_rank)
                # print(f"send activation meta: send_dtype: {send_dtype}")
        outputs: tuple[torch.Tensor] = self.pipeline.pipe_buffers["outputs"][buffer_id]
        if not self.sent_activation_meta:
            _send_activation_meta(outputs, self.next_rank)
            self.sent_activation_meta = True

        assert isinstance(outputs, tuple)
        for buffer in outputs:
            assert isinstance(buffer, torch.Tensor)
            self._send(buffer, self.next_rank)

    def recv_activations(self, buffer_id: int):
        def create_receive_buffer(sender_rank: int) -> tuple[torch.Tensor]:
            """Receive metadata about upcoming p2p transfers and return allocated buffer.

            Metadata is communicated in this order:
                * num_tensors in tensor tuple
                foreeach tensor in buffer:
                    * ndims
                    * dtype
                    * shape
                    * requires_grad
            """
            count_tensor = torch.LongTensor(data=[0]).to(self.pipeline.device)
            self._recv(count_tensor, sender_rank)
            num_tensors = count_tensor.item()
            buffers: list[torch.Tensor] = []
            for _ in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0]).to(self.pipeline.device)
                self._recv(recv_ndims, sender_rank)
                recv_ndims = recv_ndims.item()

                recv_dtype = torch.LongTensor(data=[0]).to(self.pipeline.device)
                self._recv(recv_dtype, sender_rank)
                recv_dtype = ID_TO_DTYPE[recv_dtype.item()]

                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.pipeline.device)
                self._recv(recv_shape, sender_rank)
                recv_shape = recv_shape.tolist()

                recv_req_grad = torch.LongTensor(data=[0]).to(self.pipeline.device)
                self._recv(recv_req_grad, sender_rank)
                recv_req_grad = True if recv_req_grad.item() == 1 else False
                # print(f"recv buffer: recv dtype: {recv_dtype}")
                buffers.append(
                    torch.zeros(
                        recv_shape,
                        device=self.pipeline.device,
                        dtype=recv_dtype,
                        requires_grad=recv_req_grad,
                    )
                )
            return tuple(buffers)

        if self.activation_recv_buf is None:
            self.activation_recv_buf = create_receive_buffer(self.prev_rank)

        assert isinstance(self.activation_recv_buf, tuple)
        recvd: list[torch.Tensor | None] = [None] * len(self.activation_recv_buf)
        for idx, buffer in enumerate(self.activation_recv_buf):
            assert torch.is_tensor(buffer)
            self._recv(buffer, self.prev_rank)
            recvd[idx] = buffer.clone().detach()
            recvd[idx].requires_grad = buffer.requires_grad

        self.pipeline.pipe_buffers["inputs"][buffer_id] = tuple(recvd)

    def send_gradients(self, buffer_id: int):
        inputs = self.pipeline.pipe_buffers["inputs"][buffer_id]
        assert isinstance(inputs, tuple)

        for buffer in inputs:
            # Skip tensors that will not produce a gradient
            if not buffer.requires_grad:
                assert buffer.grad is None
                continue
            assert buffer.grad is not None
            self._send(buffer.grad, self.prev_rank)

        # We can free up the input buffer now
        self.pipeline.pipe_buffers["inputs"][buffer_id] = None

    def recv_gradients(self, buffer_id: int):
        def create_gradients_buffer(
            tensors: tuple[torch.Tensor],
        ) -> tuple[torch.Tensor]:
            assert isinstance(tensors, tuple)
            buffers: list[torch.Tensor] = []
            for tensor in tensors:
                assert isinstance(tensor, torch.Tensor)
                if tensor.requires_grad:
                    buffers.append(torch.zeros_like(tensor))

            return tuple(buffers)

        outputs = self.pipeline.pipe_buffers["outputs"][buffer_id]
        assert isinstance(outputs, tuple)

        # Allocate gradients if necessary
        if self.grad_recv_buf is None:
            self.grad_recv_buf = create_gradients_buffer(outputs)

        for buffer in self.grad_recv_buf:
            self._recv(buffer, self.next_rank)


class OobleckPipeline:
    def __init__(
        self,
        pipeline_id: int,
        pipeline_template: PipelineTemplate,
        ranks: list[int],
        dataloader: OobleckDataLoader,
        step: int,
        training_args: TrainingArguments,
        is_simulated = False
    ):
        self._pipeline_id = pipeline_id
        self._template = pipeline_template
        self._ranks = ranks
        self._dataloader = dataloader
        self._global_step = step
        self._training_args = training_args
        self.device = torch.device("cuda")

        if is_simulated:
            self.my_pipeline = bool(pipeline_id == 0)
        else:
            assert dist.is_initialized(), "torch.distributed is not intialized."

            # This is used to indicate if we use this `OobleckPipeline` for training.
            self.my_pipeline = bool(dist.get_rank() in ranks)

        # Construct a 2D rank grid for this pipeline.
        # layer index -> list of ranks
        # First dimension is for layer index, second dimension is for rank.
        self.rank_grid: dict[int, list[int]] = pipeline_template.get_rank_grid(ranks)
        # print(f"this pipeline layer index -> list of ranks:{self.rank_grid}")

    def train(self):
        # A map of PipeInstruction types to methods. Each method will be executed with the
        # kwargs provided to the PipeInstruction from the scheduler.
        instruction_map = {
            schedule.OptimizerStep: self.execution.optimizer_step,
            schedule.LoadMicroBatch: self.execution.load_microbatch,
            schedule.ForwardPass: self.execution.forward_pass,
            schedule.BackwardPass: self.execution.backward_pass,
            schedule.SendActivation: self.communication.send_activations,
            schedule.RecvActivation: self.communication.recv_activations,
            schedule.SendGrad: self.communication.send_gradients,
            schedule.RecvGrad: self.communication.recv_gradients,
        }   

        for step_cmds in self.train_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in instruction_map:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                # Equivalent to: self.[execution|communication].func(buffer_id)
                instruction_map[type(cmd)](**cmd.kwargs)

        # Cleanup buffers
        for name, pipe_buffers in self.pipe_buffers.items():
            self.pipe_buffers[name] = [None] * len(pipe_buffers)

        self._global_step += 1

    def reset_iterator(self):
        self.execution._data_iterator = iter(self.execution._dataloader)

    def initialize_execution(
        self,
        model: OobleckModel,
        existing_pipeline: OobleckPipeline | None = None,
    ):
        assert self._per_layer_pgs, "Must call initialize_distributed_fsdp() first"

        layers: list[Layer] = []
        pre_stream = torch.cuda.Stream()
        post_stream = torch.cuda.Stream()
        shard_id: int = -1
        for layer_id, pg in self._per_layer_pgs.items():
            # pg:同一layer_id的不同shard组成的pg
            id = torch.distributed.get_rank(pg)
            if id < 0:
                continue

            shard_id = id
            if existing_pipeline is not None:
                existing_layer = next(
                    (
                        layer
                        for layer in existing_pipeline.execution._layers
                        if layer.layer_id == layer_id
                    ),
                    None,
                )
                if existing_layer is not None:
                    layers.append(Layer.create_layer_from_layer(existing_layer, pg))
                    continue

            layers.append(
                Layer(layer_id, model.layers[layer_id], pg, pre_stream, post_stream)
            )

        self.execution = PipelineExecution(
            pipeline=self,
            layers=layers,
            shard_id=shard_id,
            dataloader=self._dataloader,
            training_args=self._training_args,
        )

        # initialize_execution assumes to be called only if this rank is involved in
        # the pipeline. Failure of getting my_layer_index cannot happen.
        my_rank = dist.get_rank()
        my_layer_index = next(
            layer_index
            for layer_index, ranks in self.rank_grid.items()
            if my_rank in ranks
        )
        my_stage_index = next(
            stage_index
            for stage_index, stage in enumerate(self._template.get_stages())
            if my_layer_index in stage._layer_indices
        )

        sampler: OobleckSampler = self._dataloader.batch_sampler
        self.train_schedule = OobleckPipelineSchedule(
            micro_batches=sampler.num_microbatches[self._pipeline_id],
            stages=len(self._template.get_stages()),
            stage_id=my_stage_index,
        )

        num_pipe_buffers = self.train_schedule.num_pipe_buffers()
        self.pipe_buffers: dict[str, list[tuple[torch.Tensor] | None]] = {
            # batch input and received activations
            "inputs": [None for _ in range(num_pipe_buffers)],
            # labels from batch input
            "labels": [None for _ in range(num_pipe_buffers)],
            # activations to be sent
            "outputs": [None for _ in range(num_pipe_buffers)],
        }
    
    def initialize_distributed_fsdp(self):
        """Initialize torch.distributed.process_groups per layer.
        Even I am not involved in a group, torch.distributed requires all ranks to call
        `new_group()`. Thus this method should be called by everyone.

        Plus, if this rank is involved in a group, initialize execution.
        """
        # 持有同一个layer的不同shard组成的pg
        self._per_layer_pgs: dict[int, ProcessGroup] = {}
        self.execution: PipelineExecution | None = None

        for layer_id, ranks in self.rank_grid.items():
            # Remove potential duplicates
            # 即使本rank并不是new_group中的rank，pytorch也要求它调用new_group函数。并且所有人的创建顺序要相同
            pg = dist.new_group(list(set(ranks)))
            self._per_layer_pgs[layer_id] = pg

        # self.execution may not be initialized at this moment. Don't add assertion here.

    def initialize_distributed_pipeline(self):
        """Initialize torch.distributed.process_groups for a FSDP sharded pipeline.
        Even I am not involved in a group, torch.distributed requires all ranks to call
        `new_group()`. Thus this method should be called by everyone.

        Plus, if this rank is involved in a group, initialize communication.
        """
        self._per_sharded_pp_pgs: dict[int, ProcessGroup] = {}
        self.communication: PipelineCommunication | None = None

        my_rank = dist.get_rank()
        for shard_id in range(len(self.rank_grid[0])):
            ranks: list[int] = [
                ranks_per_layer[shard_id] for ranks_per_layer in self.rank_grid.values()
            ]
            # Remove potential duplicates
            pg = dist.new_group(list(set(ranks)))
            # 持有同一个shrad_id的不同layer的rank组成的pg
            self._per_sharded_pp_pgs[shard_id] = pg
            # print(f"ranks: {}")
            if my_rank in ranks:
                unique_ranks = list(set(ranks))
                rank_index = unique_ranks.index(my_rank)
                self.communication = PipelineCommunication(
                    pipeline=self,
                    process_group=pg,
                    prev_rank=unique_ranks[rank_index - 1] if rank_index > 0 else None,
                    next_rank=unique_ranks[rank_index + 1]
                    if rank_index < len(unique_ranks) - 1
                    else None,
                )

        assert len(self._per_sharded_pp_pgs) == len(
            self.rank_grid[0]
        ), "Number of per-shard process groups and model layers must match."

        # self.communication may not be initialized at this moment. Don't add assertion here.

    def is_first_stage(self) -> bool:
        return self.communication.prev_rank is None

    def is_last_stage(self) -> bool:
        return self.communication.next_rank is None
