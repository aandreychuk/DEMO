"""Fast DMM with the original consensus/message-passing semantics.

The observation encoder emits 25 spatial tokens and 13 neighbor tokens.  They
are never converted to the legacy 32-token interface.  Each communication
round gathers the same per-neighbor dynamic messages as DMMv2, then a learned
action/message query attends to all 38 observation tokens and 13 messages.
The expensive static key/value projections are computed once per MAPF step
and reused by every round.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_ACTIONS = 5


class FusionFriendlyResidualConvBlock(nn.Module):
    """ConvNeXt-like spatial block arranged for Inductor fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.norm = nn.RMSNorm(channels)
        self.expand = nn.Linear(channels, 2 * channels, bias=False)
        self.project = nn.Linear(channels, channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        value, gate = self.expand(self.norm(x)).chunk(2, dim=-1)
        x = self.project(value * F.silu(gate)).permute(0, 3, 1, 2)
        return residual + x


class ExplicitSDPASelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.onnx_explicit_attention = False
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens).view(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        allowed = (~padding_mask).unsqueeze(1).unsqueeze(1)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.onnx_explicit_attention:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.head_width**-0.5
            scores = scores.masked_fill(~allowed, float("-inf"))
            attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        else:
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=0.0
            )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(attended)


class RelationalMixerBlock(nn.Module):
    """Mix compact spatial and neighbor tokens before communication."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.attention = ExplicitSDPASelfAttention(width, heads)
        self.ff_norm = nn.RMSNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width, bias=False),
            nn.SiLU(),
            nn.Linear(2 * width, width, bias=False),
        )

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        tokens = tokens + self.attention(
            self.attention_norm(tokens), padding_mask
        )
        tokens = tokens + self.ff(self.ff_norm(tokens))
        return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


@dataclass
class FastDMMConfig:
    block_size: int = 256
    vocab_size: int = 67
    field_of_view_size: int = 121
    agent_info_size: int = 10
    max_num_neighbors: int = 13
    empty_token_code: int = 66
    empty_connection_code: int = -1

    width: int = 96
    heads: int = 4
    conv_blocks: int = 3
    mixer_blocks: int = 2
    spatial_size: int = 5
    communication_query_blocks: int = 2
    communication_hidden_multiplier: int = 2
    dropout: float = 0.0
    bias: bool = False

    dt: float = 0.25
    tau: float = 1.0
    n_comm_rounds: int = 4

    # Original DMM supervised-training controls.  Keeping these fields and
    # the forward signature identical lets the original trainer drive this
    # architecture without a second training implementation.
    dirichlet_tf_on: bool = True
    dirichlet_tf_beta: float = 1.0
    dirichlet_tf_beta_final: float = 0.8
    dirichlet_tf_anneal_steps: int = 100_000
    round_tf_on: bool = True
    round_tf_beta: float = 1.0
    round_tf_beta_final: float = 0.8
    round_tf_anneal_steps: int = 100_000


class FastStructuredEncoder(nn.Module):
    """Encode an observation as exactly 25 spatial + 13 neighbor tokens."""

    def __init__(self, config: FastDMMConfig) -> None:
        super().__init__()
        if config.block_size != 256:
            raise ValueError("FastStructuredEncoder requires 256 input tokens")
        if config.field_of_view_size != 121:
            raise ValueError("FastStructuredEncoder requires an 11x11 grid")
        if config.max_num_neighbors != 13 or config.agent_info_size != 10:
            raise ValueError("FastStructuredEncoder requires 13x10 neighbor data")
        if config.spatial_size != 5:
            raise ValueError("The scratch architecture uses a 5x5 spatial grid")

        width = config.width
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, width)
        self.grid_stem = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.grid_blocks = nn.Sequential(
            *(FusionFriendlyResidualConvBlock(width) for _ in range(config.conv_blocks))
        )
        self.neighbor_mlp = nn.Sequential(
            nn.Linear(config.agent_info_size * width, 2 * width, bias=False),
            nn.SiLU(),
            nn.Linear(2 * width, width, bias=False),
        )
        self.spatial_positions = nn.Parameter(torch.randn(25, width) * 0.02)
        self.neighbor_slots = nn.Parameter(torch.randn(13, width) * 0.02)
        self.token_types = nn.Parameter(torch.randn(2, width) * 0.02)
        self.mixer = nn.ModuleList(
            RelationalMixerBlock(width, config.heads)
            for _ in range(config.mixer_blocks)
        )
        self.output_norm = nn.RMSNorm(width)

    def forward(
        self, observations: Tensor, neighbor_padding: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if observations.shape[-1] != 256:
            raise ValueError(f"Expected 256 tokens, got {observations.shape}")
        flat = observations.reshape(-1, 256)
        width = self.config.width

        grid = self.token_embedding(flat[:, :121])
        grid = grid.transpose(1, 2).reshape(-1, width, 11, 11)
        grid = self.grid_blocks(self.grid_stem(grid))
        grid = F.adaptive_avg_pool2d(grid, (5, 5)).flatten(2).transpose(1, 2)
        grid = grid + self.spatial_positions.unsqueeze(0) + self.token_types[0]

        neighbor_ids = flat[:, 121:251].reshape(-1, 13, 10)
        if neighbor_padding is None:
            neighbor_padding = neighbor_ids.eq(self.config.empty_token_code).all(-1)
        else:
            neighbor_padding = neighbor_padding.reshape(-1, 13)
        neighbors = self.token_embedding(neighbor_ids).flatten(2)
        neighbors = self.neighbor_mlp(neighbors)
        neighbors = neighbors + self.neighbor_slots.unsqueeze(0) + self.token_types[1]

        tokens = torch.cat((grid, neighbors), dim=1)
        spatial_padding = torch.zeros(
            (flat.shape[0], 25), dtype=torch.bool, device=flat.device
        )
        padding = torch.cat((spatial_padding, neighbor_padding), dim=1)
        for block in self.mixer:
            tokens = block(tokens, padding)
        tokens = self.output_norm(tokens)
        tokens = tokens.masked_fill(padding.unsqueeze(-1), 0.0)
        return tokens, padding


class QueryCrossAttentionBlock(nn.Module):
    """One query-only cross-attention/SwiGLU block with its own K/V space."""

    def __init__(self, config: FastDMMConfig) -> None:
        super().__init__()
        width = config.width
        hidden = config.communication_hidden_multiplier * width
        self.heads = config.heads
        self.head_width = width // config.heads
        self.onnx_explicit_attention = False
        self.memory_norm = nn.RMSNorm(width)
        self.static_key_value = nn.Linear(width, 2 * width, bias=False)
        self.message_key_value = nn.Linear(width, 2 * width, bias=False)
        self.query_norm = nn.RMSNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.attention_output_norm = nn.RMSNorm(width)
        self.ff_expand = nn.Linear(width, 2 * hidden, bias=False)
        self.ff_project = nn.Linear(hidden, width, bias=False)
        self.output_norm = nn.RMSNorm(width)

    def prepare_static_memory(
        self, observation_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_agents, token_count, width = observation_tokens.shape
        key_value = self.static_key_value(
            self.memory_norm(observation_tokens)
        ).view(
            batch_agents,
            token_count,
            2,
            self.heads,
            self.head_width,
        )
        key, value = key_value.unbind(dim=2)
        return key.transpose(1, 2), value.transpose(1, 2)

    def forward(
        self,
        query_token: Tensor,
        messages: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        allowed: Tensor,
    ) -> Tensor:
        batch_agents, message_count, width = messages.shape
        dynamic_key_value = self.message_key_value(
            self.memory_norm(messages)
        ).view(
            batch_agents,
            message_count,
            2,
            self.heads,
            self.head_width,
        )
        dynamic_key, dynamic_value = dynamic_key_value.unbind(dim=2)
        key = torch.cat((static_key, dynamic_key.transpose(1, 2)), dim=2)
        value = torch.cat((static_value, dynamic_value.transpose(1, 2)), dim=2)
        query = self.query(self.query_norm(query_token)).view(
            batch_agents, 1, self.heads, self.head_width
        ).transpose(1, 2)
        if self.onnx_explicit_attention:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.head_width**-0.5
            scores = scores.masked_fill(~allowed, float("-inf"))
            attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        else:
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=0.0
            )
        attended = attended.transpose(1, 2).reshape(batch_agents, 1, width)
        feature = query_token + self.output(attended)
        normalized = self.attention_output_norm(feature)
        ff_value, ff_gate = self.ff_expand(normalized).chunk(2, dim=-1)
        feature = feature + self.ff_project(ff_value * F.silu(ff_gate))
        return self.output_norm(feature)


class CachedCrossAttentionCommunication(nn.Module):
    """DMM communication decoder with cached observation key/value tensors.

    This preserves the legacy round structure:

    ``h + z -> gather neighbor slots -> shared action/message feature -> logits,h``.

    Only the mechanism used to read local latents and messages changes.  A
    a small query stack uses fused SDPA over 38 static observation tokens and
    up to 13 fresh message tokens instead of repeatedly running full
    self-attention over their concatenation.
    """

    def __init__(self, config: FastDMMConfig) -> None:
        super().__init__()
        width = config.width
        if width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.max_num_neighbors = config.max_num_neighbors
        self.empty_message = nn.Parameter(torch.randn(width) * 0.02)
        self.slot_embedding = nn.Parameter(
            torch.randn(config.max_num_neighbors, width) * 0.02
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        if config.communication_query_blocks < 1:
            raise ValueError("communication_query_blocks must be positive")
        self.blocks = nn.ModuleList(
            QueryCrossAttentionBlock(config)
            for _ in range(config.communication_query_blocks)
        )

    @staticmethod
    def collect(sender: Tensor, connections: Tensor, empty: Tensor) -> Tensor:
        """Gather sender states; connection -1 maps to a learned empty row."""
        batch, agents, width = sender.shape
        _, _, neighbors = connections.shape
        source = torch.cat(
            (empty.reshape(1, 1, width).expand(batch, 1, width), sender),
            dim=1,
        )
        indices = (connections.long() + 1).unsqueeze(-1).expand(
            batch, agents, neighbors, width
        )
        return torch.gather(
            source.unsqueeze(2).expand(batch, agents + 1, neighbors, width),
            dim=1,
            index=indices,
        )

    def prepare_static_memory(
        self, observation_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Project 38 static tokens once for every query block.

        The leading dimension indexes blocks and remains outside the four-round
        communication loop.
        """
        projected = [
            block.prepare_static_memory(observation_tokens)
            for block in self.blocks
        ]
        return (
            torch.stack([key for key, _ in projected]),
            torch.stack([value for _, value in projected]),
        )

    def forward(
        self,
        agent_to_message: Tensor,
        connections: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
    ) -> Tensor:
        batch, agents, width = agent_to_message.shape
        messages = self.collect(
            agent_to_message, connections, self.empty_message
        )
        return self.forward_collected(
            messages,
            connections,
            static_key,
            static_value,
            observation_padding,
        )

    def forward_collected(
        self,
        messages: Tensor,
        connections: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
    ) -> Tensor:
        """Decode receiver messages already gathered from the global table."""
        batch, agents, neighbors, width = messages.shape
        batch_agents = batch * agents
        valid = connections.ge(0)
        messages = messages + self.slot_embedding[: connections.shape[-1]]
        messages = messages.reshape(batch_agents, neighbors, width)
        query_token = self.query_token.expand(batch_agents, 1, width)
        padding = torch.cat(
            (observation_padding, ~valid.reshape(batch_agents, -1)), dim=1
        )
        allowed = (~padding).unsqueeze(1).unsqueeze(1)
        for block_index, block in enumerate(self.blocks):
            query_token = block(
                query_token,
                messages,
                static_key[block_index],
                static_value[block_index],
                allowed,
            )
        return query_token[:, 0]


class FastDMM(nn.Module):
    """Scratch DMM with a structured encoder and lightweight GNN rounds."""

    def __init__(self, config: FastDMMConfig) -> None:
        super().__init__()
        self.config = config
        self.num_actions = NUM_ACTIONS
        self.representation_encoder = FastStructuredEncoder(config)
        self.z_proj = nn.Linear(NUM_ACTIONS, config.width, bias=False)
        self.communication = CachedCrossAttentionCommunication(config)
        self.pi_head = nn.Linear(config.width, NUM_ACTIONS, bias=False)
        self.msg_head = nn.Linear(config.width, config.width, bias=False)
        self.onnx_simplify_consensus = False

    def _centred_log_ohe(self, actions: Tensor, dtype: torch.dtype) -> Tensor:
        if not self.onnx_simplify_consensus:
            one_hot = F.one_hot(actions, NUM_ACTIONS).to(dtype)
            target = F.log_softmax((one_hot + 1e-8).log(), dim=-1)
            return target - target.mean(-1, keepdim=True)
        # ``log_softmax(log(one_hot + eps))`` followed by centering is exactly
        # ``log(one_hot + eps) - mean(log(one_hot + eps))``: the shared
        # log-sum-exp term cancels.  Express the two possible values directly
        # so deployment exporters do not materialize OneHot/Log/LogSoftmax
        # subgraphs (and cannot fold away the epsilon before Log).
        epsilon = 1e-8
        low = torch.log(torch.tensor(epsilon, dtype=dtype, device=actions.device))
        high = torch.log(
            torch.tensor(1.0 + epsilon, dtype=dtype, device=actions.device)
        )
        mean = (high + (NUM_ACTIONS - 1) * low) / NUM_ACTIONS
        values = torch.arange(NUM_ACTIONS, device=actions.device)
        return torch.where(
            actions.unsqueeze(-1).eq(values), high - mean, low - mean
        )

    @staticmethod
    def _dirichlet_z0(
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        alpha: float,
        targets: Tensor | None = None,
        teacher_forcing_beta: float = 0.0,
    ) -> Tensor:
        concentration = torch.full((count, NUM_ACTIONS), alpha)
        if targets is not None and teacher_forcing_beta > 0.0:
            valid = targets.ne(-1).cpu()
            safe_targets = targets.clamp(min=0).cpu()
            use_teacher = torch.bernoulli(
                torch.full((count,), teacher_forcing_beta)
            ).bool() & valid
            target_concentration = F.one_hot(
                safe_targets, NUM_ACTIONS
            ).float()
            concentration[use_teacher] += target_concentration[use_teacher]
        probabilities = torch.distributions.Dirichlet(concentration).sample()
        z = probabilities.clamp(min=1e-8).log()
        z = z - z.mean(-1, keepdim=True)
        return z.to(device=device, dtype=dtype)

    @staticmethod
    def _prob_to_log_centered(probabilities: Tensor) -> Tensor:
        z = probabilities.float().clamp(min=1e-8).log()
        return (z - z.mean(-1, keepdim=True)).to(probabilities.dtype)

    def _encode(
        self, obs: Tensor, neighbor_padding: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, agents, token_count = obs.shape
        tokens, padding = self.representation_encoder(
            obs.reshape(batch * agents, token_count), neighbor_padding
        )
        static_key, static_value = self.communication.prepare_static_memory(tokens)
        return static_key, static_value, padding, tokens

    def _run_round(
        self,
        z: Tensor,
        h: Tensor,
        agent_chat_ids: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, agents, _ = agent_chat_ids.shape
        width = h.shape[-1]
        agent_to_message = h + self.z_proj(z).reshape(
            batch, agents, width
        ).to(h.dtype)
        feature = self.communication(
            agent_to_message,
            agent_chat_ids,
            static_key,
            static_value,
            observation_padding,
        )
        logits = self.pi_head(feature)
        h_new = self.msg_head(feature).reshape(batch, agents, width)
        return logits, h_new

    def _encode_chunked(
        self, obs: Tensor, agent_chunk_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode once while bounding temporary activations by agent chunk size."""
        batch, agents, token_count = obs.shape
        count = batch * agents
        chunk = min(max(1, int(agent_chunk_size)), count)
        flat = obs.reshape(count, token_count)
        static_key = static_value = padding = None
        for start in range(0, count, chunk):
            end = min(start + chunk, count)
            tokens, current_padding = self.representation_encoder(flat[start:end])
            current_key, current_value = self.communication.prepare_static_memory(tokens)
            if static_key is None:
                static_key = torch.empty(
                    (current_key.shape[0], count, *current_key.shape[2:]),
                    dtype=current_key.dtype,
                    device=current_key.device,
                )
                static_value = torch.empty_like(static_key)
                padding = torch.empty(
                    (count, current_padding.shape[1]),
                    dtype=torch.bool,
                    device=current_padding.device,
                )
            static_key[:, start:end] = current_key
            static_value[:, start:end] = current_value
            padding[start:end] = current_padding
        assert static_key is not None and static_value is not None and padding is not None
        return static_key, static_value, padding

    @staticmethod
    def _collect_message_chunk(
        sender: Tensor, connections: Tensor, empty: Tensor
    ) -> Tensor:
        """Gather [B,C,L,W] without expanding sender across all receivers."""
        batch, _, width = sender.shape
        receivers, neighbors = connections.shape[1:]
        source = torch.cat(
            (empty.reshape(1, 1, width).expand(batch, 1, width), sender), dim=1
        )
        indices = (connections.long() + 1).reshape(
            batch, receivers * neighbors, 1
        ).expand(-1, -1, width)
        return torch.gather(source, 1, indices).reshape(
            batch, receivers, neighbors, width
        )

    def _run_round_chunked(
        self,
        z: Tensor,
        h: Tensor,
        agent_chat_ids: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
        agent_chunk_size: int,
    ) -> tuple[Tensor, Tensor]:
        batch, agents, _ = agent_chat_ids.shape
        width = h.shape[-1]
        chunk = min(max(1, int(agent_chunk_size)), agents)
        sender = h + self.z_proj(z).reshape(batch, agents, width).to(h.dtype)
        logits = torch.empty(
            (batch, agents, NUM_ACTIONS), dtype=h.dtype, device=h.device
        )
        h_new = torch.empty_like(h)
        key_view = static_key.reshape(
            static_key.shape[0], batch, agents, *static_key.shape[2:]
        )
        value_view = static_value.reshape(
            static_value.shape[0], batch, agents, *static_value.shape[2:]
        )
        padding_view = observation_padding.reshape(batch, agents, -1)
        for start in range(0, agents, chunk):
            end = min(start + chunk, agents)
            connections = agent_chat_ids[:, start:end]
            messages = self._collect_message_chunk(
                sender, connections, self.communication.empty_message
            )
            key = key_view[:, :, start:end].reshape(
                static_key.shape[0], batch * (end - start), *static_key.shape[2:]
            )
            value = value_view[:, :, start:end].reshape(
                static_value.shape[0], batch * (end - start), *static_value.shape[2:]
            )
            padding = padding_view[:, start:end].reshape(batch * (end - start), -1)
            feature = self.communication.forward_collected(
                messages, connections, key, value, padding
            )
            logits[:, start:end] = self.pi_head(feature).reshape(
                batch, end - start, NUM_ACTIONS
            )
            h_new[:, start:end] = self.msg_head(feature).reshape(
                batch, end - start, width
            )
        return logits.reshape(batch * agents, NUM_ACTIONS), h_new

    def _run_sharded_round_chunked(
        self,
        z: Tensor,
        h: Tensor,
        agent_chat_ids: Tensor,
        static_key: Tensor,
        static_value: Tensor,
        observation_padding: Tensor,
        agent_chunk_size: int,
        gather_sender,
    ) -> tuple[Tensor, Tensor]:
        """Run local receivers after gathering global dynamic messages."""
        batch, local_agents, _ = agent_chat_ids.shape
        width = h.shape[-1]
        chunk = min(max(1, int(agent_chunk_size)), local_agents)
        local_sender = h + self.z_proj(z).reshape(
            batch, local_agents, width
        ).to(h.dtype)
        global_sender = gather_sender(local_sender.contiguous())
        logits = torch.empty(
            (batch, local_agents, NUM_ACTIONS), dtype=h.dtype, device=h.device
        )
        h_new = torch.empty_like(h)
        key_view = static_key.reshape(
            static_key.shape[0], batch, local_agents, *static_key.shape[2:]
        )
        value_view = static_value.reshape(
            static_value.shape[0], batch, local_agents, *static_value.shape[2:]
        )
        padding_view = observation_padding.reshape(batch, local_agents, -1)
        for start in range(0, local_agents, chunk):
            end = min(start + chunk, local_agents)
            connections = agent_chat_ids[:, start:end]
            messages = self._collect_message_chunk(
                global_sender, connections, self.communication.empty_message
            )
            key = key_view[:, :, start:end].reshape(
                static_key.shape[0], batch * (end - start), *static_key.shape[2:]
            )
            value = value_view[:, :, start:end].reshape(
                static_value.shape[0], batch * (end - start), *static_value.shape[2:]
            )
            padding = padding_view[:, start:end].reshape(batch * (end - start), -1)
            feature = self.communication.forward_collected(
                messages, connections, key, value, padding
            )
            logits[:, start:end] = self.pi_head(feature).reshape(
                batch, end - start, NUM_ACTIONS
            )
            h_new[:, start:end] = self.msg_head(feature).reshape(
                batch, end - start, width
            )
        return logits.reshape(batch * local_agents, NUM_ACTIONS), h_new

    def _run_sharded_active_round_chunked(
        self,
        z: Tensor,
        h: Tensor,
        active_ids: Tensor,
        active_chat_ids: Tensor,
        static_key: Tensor | None,
        static_value: Tensor | None,
        observation_padding: Tensor | None,
        agent_chunk_size: int,
        gather_sender,
    ) -> tuple[Tensor, Tensor]:
        """Run only active local receivers while gathering full-shard senders."""
        batch, local_agents, width = h.shape
        if batch != 1:
            raise ValueError("active sharded inference currently requires batch size 1")
        active_count = active_ids.numel()
        local_sender = h + self.z_proj(z).reshape(
            batch, local_agents, width
        ).to(h.dtype)
        global_sender = gather_sender(local_sender.contiguous())
        logits = torch.empty(
            (active_count, NUM_ACTIONS), dtype=h.dtype, device=h.device
        )
        h_new = h.clone()
        if not active_count:
            return logits, h_new
        assert static_key is not None and static_value is not None
        assert observation_padding is not None
        chunk = min(max(1, int(agent_chunk_size)), active_count)
        for start in range(0, active_count, chunk):
            end = min(start + chunk, active_count)
            connections = active_chat_ids[:, start:end]
            messages = self._collect_message_chunk(
                global_sender, connections, self.communication.empty_message
            )
            key = static_key[:, start:end]
            value = static_value[:, start:end]
            padding = observation_padding[start:end]
            feature = self.communication.forward_collected(
                messages, connections, key, value, padding
            )
            logits[start:end] = self.pi_head(feature).reshape(
                end - start, NUM_ACTIONS
            )
            active_h = self.msg_head(feature).reshape(1, end - start, width)
            h_new[:, active_ids[start:end]] = active_h.to(h_new.dtype)
        return logits, h_new

    @torch.no_grad()
    def deterministic_zero_act(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        agent_chunk_size: int = 0,
        neighbor_padding: Tensor | None = None,
    ) -> Tensor:
        """Deployment path: z0=0 and argmax votes in all communication rounds."""
        batch, agents, _ = obs.shape
        count = batch * agents
        if agent_chunk_size > 0:
            static_key, static_value, padding = self._encode_chunked(
                obs, agent_chunk_size
            )
        else:
            static_key, static_value, padding, _ = self._encode(
                obs, neighbor_padding
            )
        dtype = static_key.dtype
        z = torch.zeros(count, NUM_ACTIONS, dtype=dtype, device=obs.device)
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        for _ in range(self.config.n_comm_rounds):
            if agent_chunk_size > 0:
                logits, h = self._run_round_chunked(
                    z, h, agent_chat_ids, static_key, static_value, padding,
                    agent_chunk_size,
                )
            else:
                logits, h = self._run_round(
                    z, h, agent_chat_ids, static_key, static_value, padding
                )
            if self.onnx_simplify_consensus:
                # Preserve nan_to_num(NaN=0) semantics without IsNaN/IsInf
                # fallback nodes. NaN is the only floating-point value that
                # does not compare equal to itself; Equal and Where both stay
                # on the WebGPU execution provider.
                logits = torch.where(logits.eq(logits), logits, 0.0)
            else:
                logits = logits.nan_to_num(0.0)
            vote = logits.argmax(dim=-1)
            target = self._centred_log_ohe(vote, dtype)
            z = z + self.config.dt * (target - z)
        # The first centred-log update promotes z to FP32 under BF16 autocast.
        # Avoid a redundant terminal cast: Torch 2.13 records conflicting
        # autocast metadata for it when the exported program is lowered again
        # by AOTAutograd.
        return z.reshape(batch, agents, NUM_ACTIONS)

    @torch.no_grad()
    def deterministic_zero_act_sharded(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        gather_sender,
        agent_chunk_size: int,
        active_mask: Tensor | None = None,
    ) -> Tensor:
        """Exact deterministic path for local receivers with global messages."""
        batch, local_agents, _ = obs.shape
        count = batch * local_agents
        if active_mask is not None:
            if batch != 1:
                raise ValueError(
                    "active sharded inference currently requires batch size 1"
                )
            active_mask = active_mask.to(device=obs.device, dtype=torch.bool)
            if active_mask.shape != (local_agents,):
                raise ValueError("active_mask must have shape [local_agents]")
            active_ids = active_mask.nonzero(as_tuple=False).flatten()
            if active_ids.numel():
                static_key, static_value, padding = self._encode_chunked(
                    obs[:, active_ids], agent_chunk_size
                )
                dtype = static_key.dtype
            else:
                static_key = static_value = padding = None
                if obs.is_cuda and torch.is_autocast_enabled("cuda"):
                    dtype = torch.get_autocast_dtype("cuda")
                else:
                    dtype = self.communication.empty_message.dtype
            z = torch.zeros(count, NUM_ACTIONS, dtype=dtype, device=obs.device)
            h = self.communication.empty_message.reshape(1, 1, -1).expand(
                batch, local_agents, -1
            ).to(dtype)
            active_chat_ids = agent_chat_ids[:, active_ids]
            for _ in range(self.config.n_comm_rounds):
                logits, h = self._run_sharded_active_round_chunked(
                    z, h, active_ids, active_chat_ids, static_key, static_value,
                    padding, agent_chunk_size, gather_sender,
                )
                vote = logits.nan_to_num(0.0).argmax(dim=-1)
                target = self._centred_log_ohe(vote, dtype)
                updated_z = (
                    z[active_ids]
                    + self.config.dt * (target - z[active_ids])
                )
                # CUDA autocast may promote log_softmax (and therefore the
                # consensus update) to fp32.  The full path promotes ``z``
                # implicitly when it rebinds the whole tensor; mirror that
                # behavior before the indexed active-agent assignment.
                if updated_z.dtype != z.dtype:
                    z = z.to(updated_z.dtype)
                z[active_ids] = updated_z
            return z.reshape(batch, local_agents, NUM_ACTIONS).float()

        static_key, static_value, padding = self._encode_chunked(
            obs, agent_chunk_size
        )
        dtype = static_key.dtype
        z = torch.zeros(count, NUM_ACTIONS, dtype=dtype, device=obs.device)
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, local_agents, -1
        ).to(dtype)
        for _ in range(self.config.n_comm_rounds):
            logits, h = self._run_sharded_round_chunked(
                z, h, agent_chat_ids, static_key, static_value, padding,
                agent_chunk_size, gather_sender,
            )
            vote = logits.nan_to_num(0.0).argmax(dim=-1)
            target = self._centred_log_ohe(vote, dtype)
            z = z + self.config.dt * (target - z)
        return z.reshape(batch, local_agents, NUM_ACTIONS).float()

    @torch.no_grad()
    def stochastic_act_sharded(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        gather_sender,
        agent_chunk_size: int,
        generator: torch.Generator,
        active_mask: Tensor | None = None,
        dir_alpha: float = 1.0,
        rollout_tau: float = 1.0,
    ) -> Tensor:
        """Sample native consensus only for active sharded receivers.

        Inactive agents retain the empty sender message and a zero consensus
        vector, so their policy proposal is wait.  Every rank still performs
        the same four sender gathers even when it has no active receivers.
        """
        batch, local_agents, _ = obs.shape
        if batch != 1:
            raise ValueError("stochastic sharded inference requires batch size 1")
        if active_mask is None:
            active_mask = torch.ones(
                local_agents, dtype=torch.bool, device=obs.device
            )
        else:
            active_mask = active_mask.to(device=obs.device, dtype=torch.bool)
            if active_mask.shape != (local_agents,):
                raise ValueError("active_mask must have shape [local_agents]")
        active_ids = active_mask.nonzero(as_tuple=False).flatten()
        if active_ids.numel():
            static_key, static_value, padding = self._encode_chunked(
                obs[:, active_ids], agent_chunk_size
            )
            dtype = static_key.dtype
        else:
            static_key = static_value = padding = None
            if obs.is_cuda and torch.is_autocast_enabled("cuda"):
                dtype = torch.get_autocast_dtype("cuda")
            else:
                dtype = self.communication.empty_message.dtype

        count = local_agents
        concentration = torch.full(
            (count, NUM_ACTIONS),
            dir_alpha,
            dtype=torch.float32,
            device=obs.device,
        )
        gamma = torch._standard_gamma(concentration, generator=generator)
        probabilities = gamma / gamma.sum(dim=-1, keepdim=True)
        sampled_z0 = probabilities.clamp_min(1e-8).log()
        sampled_z0 -= sampled_z0.mean(dim=-1, keepdim=True)
        z = torch.zeros(count, NUM_ACTIONS, dtype=dtype, device=obs.device)
        z[active_ids] = sampled_z0[active_ids].to(dtype)
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, local_agents, -1
        ).to(dtype)
        active_chat_ids = agent_chat_ids[:, active_ids]
        for _ in range(self.config.n_comm_rounds):
            logits, h = self._run_sharded_active_round_chunked(
                z,
                h,
                active_ids,
                active_chat_ids,
                static_key,
                static_value,
                padding,
                agent_chunk_size,
                gather_sender,
            )
            action_probabilities = torch.softmax(
                logits.float() / rollout_tau, dim=-1
            )
            uniforms = torch.rand(
                count, generator=generator, device=obs.device
            )[active_ids]
            vote = (
                (uniforms[:, None] > action_probabilities.cumsum(dim=-1))
                .sum(dim=-1)
                .clamp_max(NUM_ACTIONS - 1)
            )
            target = self._centred_log_ohe(vote, dtype)
            updated_z = (
                z[active_ids]
                + self.config.dt * (target - z[active_ids])
            )
            if updated_z.dtype != z.dtype:
                z = z.to(updated_z.dtype)
            z[active_ids] = updated_z
        return z.reshape(batch, local_agents, NUM_ACTIONS).float()

    def forward(
        self,
        observations: Tensor,
        agent_chat_ids: Tensor,
        target_actions: Tensor,
        dirichlet_tf_beta: float | None = None,
        round_tf_beta: float | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Original DMM supervised forward contract.

        The original DMM trainer owns teacher-forcing schedules, DDP,
        optimisation, and checkpointing.  This method only implements the
        model-side per-round categorical objective expected by that trainer.
        """
        batch, agents, _ = observations.shape
        count = batch * agents
        targets = target_actions.reshape(count)
        static_key, static_value, observation_padding, _ = self._encode(
            observations
        )
        dtype = static_key.dtype

        dirichlet_beta = (
            self.config.dirichlet_tf_beta
            if dirichlet_tf_beta is None
            else dirichlet_tf_beta
        )
        if not self.config.dirichlet_tf_on:
            dirichlet_beta = 0.0
        round_beta = (
            self.config.round_tf_beta
            if round_tf_beta is None
            else round_tf_beta
        )
        if not self.config.round_tf_on:
            round_beta = 0.0

        z = self._dirichlet_z0(
            count,
            observations.device,
            dtype,
            1.0,
            targets=targets,
            teacher_forcing_beta=dirichlet_beta,
        )
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        per_round_losses: list[Tensor] = []

        for _ in range(self.config.n_comm_rounds):
            logits, h = self._run_round(
                z,
                h,
                agent_chat_ids,
                static_key,
                static_value,
                observation_padding,
            )
            per_round_losses.append(
                F.cross_entropy(logits, targets, ignore_index=-1)
            )
            with torch.no_grad():
                sampled = torch.distributions.Categorical(logits=logits).sample()
                if round_beta > 0.0:
                    use_teacher = torch.bernoulli(
                        torch.full(
                            (count,), round_beta, device=observations.device
                        )
                    ).bool() & targets.ne(-1)
                    votes = torch.where(
                        use_teacher, targets.clamp(min=0), sampled
                    )
                else:
                    votes = sampled
            target = self._centred_log_ohe(votes, dtype)
            z = z + self.config.dt * (target - z)

        return torch.stack(per_round_losses).mean(), per_round_losses

    @torch.no_grad()
    def rollout_act(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        rollout_tau: float = 1.0,
        z0_in: Tensor | None = None,
        dir_alpha: float = 1.0,
    ):
        batch, agents, _ = obs.shape
        count = batch * agents
        static_key, static_value, observation_padding, _ = self._encode(obs)
        dtype = static_key.dtype
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        if z0_in is None:
            z0 = self._dirichlet_z0(count, obs.device, dtype, dir_alpha)
        else:
            z0 = self._prob_to_log_centered(
                z0_in.to(device=obs.device, dtype=dtype)
            )
        z = z0.clone()
        votes_list = []
        log_pi_list = []
        for _ in range(self.config.n_comm_rounds):
            logits, h = self._run_round(
                z,
                h,
                agent_chat_ids,
                static_key,
                static_value,
                observation_padding,
            )
            logits = logits.nan_to_num(0.0)
            vote = torch.distributions.Categorical(
                logits=logits / rollout_tau
            ).sample()
            log_pi = F.log_softmax(logits, dim=-1).gather(
                -1, vote.unsqueeze(-1)
            ).squeeze(-1)
            votes_list.append(vote)
            log_pi_list.append(log_pi)
            target = self._centred_log_ohe(vote, dtype)
            z = z + self.config.dt * (target - z)
        return (
            z.argmax(-1),
            torch.stack(votes_list, dim=-1),
            torch.stack(log_pi_list, dim=-1),
            z0,
            z,
        )

    def forward_grpo(
        self,
        obs: Tensor,
        agent_chat_ids: Tensor,
        stored_votes: Tensor,
        stored_z0: Tensor,
    ):
        batch, agents, _ = obs.shape
        static_key, static_value, observation_padding, _ = self._encode(obs)
        dtype = static_key.dtype
        z = stored_z0.to(dtype)
        h = self.communication.empty_message.reshape(1, 1, -1).expand(
            batch, agents, -1
        ).to(dtype)
        log_pi_list = []
        logits_list = []
        for round_index in range(self.config.n_comm_rounds):
            logits, h = self._run_round(
                z,
                h,
                agent_chat_ids,
                static_key,
                static_value,
                observation_padding,
            )
            vote = stored_votes[:, round_index]
            log_pi = F.log_softmax(logits, dim=-1).gather(
                -1, vote.unsqueeze(-1)
            ).squeeze(-1)
            log_pi_list.append(log_pi)
            logits_list.append(logits)
            target = self._centred_log_ohe(vote.detach(), dtype)
            z = z + self.config.dt * (target - z)
        return torch.stack(log_pi_list, -1), torch.stack(logits_list, 1)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if fused and device_type == "cuda" else {}
        return torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas, **extra
        )
