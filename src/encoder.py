from typing import Mapping, Tuple, Sequence

from einops import rearrange, repeat
import torch
import torch.nn as nn
from transformers import PerceiverConfig
import transformers.models.perceiver.modeling_perceiver as perceiver


class PerceiverForTimeSeriesAutoencoding:
    pass

class PerceiverForTimeSeriesPrediction:
    pass

class RawSequencePreprocessor(perceiver.AbstractPreprocessor):
    pass

class DateAlignedPreprocessor(perceiver.AbstractPreprocessor):
    """
    A multimodal preprocessor for text (sentence embeddings) and
     time series (date-aligned patches of samples).
    """

    def __init__(
        self,
        config: PerceiverConfig,
        modalities: Mapping[str : perceiver.PreprocessorType],
        pos_out_channels: int,
        min_padding_size: int = 2,
        position_encoding_type: str = "fourier",
        add_or_concat_pos: str = "add",
        project_pos_dim: int = -1,
        **position_encoding_kwargs,
    ) -> None:

        self._config = config
        self._min_padding_size = min_padding_size
        self._add_or_concat_pos = add_or_concat_pos
        self._pos_out_channels = pos_out_channels

        # Create paddings.
        self._padding = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.randn(1, self.num_channels - preprocessor.num_channels)
                )
                for modality, preprocessor in modalities.items()
            }
        )

        # Create combined positional encoding.
        (
            self._position_embedding,
            self._position_projection,
        ) = perceiver.build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=pos_out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

    @property
    def num_channels(self) -> int:
        max_channel_size = max(
            processor.num_channels for _, processor in self._modalities.items()
        )
        common_channel_size = max_channel_size + self._min_padding_size
        if self._add_or_concat_pos == "concat":
            common_channel_size += self._pos_out_channels
        return common_channel_size

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> Tuple[
        Mapping[str, torch.Tensor],
        Mapping[str, Sequence[int]],
        Mapping[str, torch.Tensor],
    ]:
        """
        Args:
            inputs: dictionary with keys ("data", "metadata"), mapping to tensors
                with shape [batch, num_series, time, num_channels]. Specifically:
                    Data: [batch, num_series, k (>= 1), num_channels].
                    Metadata: [batch, num_series, 1, num_channels].
        Returns:
            final_inputs: Dictionary mapping modality name to that modality's input tensor,
                preprocessed with modality-specific and overall preprocessing. For "data",
                tensor is combination of the common (minimum) padding, modality-specific padding,
                common (series' level) positional encoding, and modality-specific (patch level)
                positional encoding. For "metadata", the combination is the same, except for
                the patch level positional encoding.
            modality_sizes: Dictionary mapping modality name to shape of the final
                input tensor for that modality.
            inputs_with_pos: Dictionary mapping modality name to modality's input tensor,
                with modality specific pre-processing performed.
        """
        # Build overall position encoding.
        batch_size, num_series, _, _ = inputs["data"]
        pos_enc = self._position_embedding(batch_size, num_series)

        # Preprocess and pad.
        padded = {}
        modality_sizes = {}
        inputs_without_pos = {}

        for modality, preprocessor in self._modalities.items():
            # Preprocess each modality using its respective preprocessor.
            output, _, inputs_without_pos[modality] = preprocessor(inputs[modality])

            # Apply common position encoding.
            _, _, time, _ = output.shape
            pos_enc = repeat(pos_enc, "b n c -> b n t c", t=time)

            if self._add_or_concat_pos == "add":
                output_with_pos = output + pos_enc
            elif self._add_or_concat_pos == "concat":
                output_with_pos = torch.cat([output_with_pos, padding], dim=-1)

            # Pad to common channel size.
            padding = repeat(
                self._padding[modality],
                "1 c -> b n t c",
                b=batch_size,
                n=num_series,
                t=time,
            )
            output_padded = torch.cat([output_with_pos, padding], dim=-1)

            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape

        # Apply a predictable ordering to the modalities
        padded_ls = [padding[k] for k in sorted(padded.keys())]

        # Concatenate along time dimension.
        final_inputs = torch.cat(padded_ls, dim=2)

        return final_inputs, modality_sizes, inputs_without_pos


class DataPreprocessor(perceiver.AbstractPreprocessor):
    def __init__(
        self,
        config: PerceiverConfig,
        num_patches_per_series: int,
        num_samples_per_patch: int,
        out_channels: int,
        project_pos_dim: int = -1,
        position_encoding_type: str = "fourier",
        concat_or_add_pos: str = "concat",
        normalize_windows: bool = False,
        **position_encoding_kwargs,
    ) -> None:

        self._config = config
        self._normalize_windows = normalize_windows
        self._concat_or_add_pos = concat_or_add_pos
        self._num_patches_per_series = num_patches_per_series
        self._num_samples_per_patch = num_samples_per_patch
        self._position_encoding_type = position_encoding_type
        self._project_pos_dim = project_pos_dim

        # Build position embeddings.
        (
            self._position_embeddings,
            self._positions_projection,
        ) = perceiver.build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

    @property
    def num_channels(self) -> int:
        if self._project_pos_dim > 0:
            pos_dim = self._project_pos_dim
        else:
            pos_dim = self._position_embeddings.output_size()

        if self._concat_or_add_pos == "add":
            return pos_dim

        return self._num_samples_per_patch + pos_dim

    def forward(self, inputs: torch.Tensor) -> torch.tensor:
        """
        Args:
            inputs: tensor of size [batch, num_series, num_samples_per_series]
        """
        # Determine input dimensions.
        batch_size = inputs.shape[0]
        num_series = inputs.shape[1]
        num_samples_per_series = inputs.shape[2]

        if num_samples_per_series != (
            self._num_patches_per_series * self._num_samples_per_patch
        ):
            # raise exception.
            pass

        # Construct the position encoding.
        if self._position_encoding_type == "trainable":
            pos_enc = self._position_embeddings(batch_size=batch_size)
        elif self._position_encoding_type == "fourier":
            if self._num_patches_per_series > 1:
                pos_enc = self._position_embeddings(
                    index_dims=self._num_patches_per_series,
                    batch_size=batch_size,
                    device=inputs.device,
                )
            else:
                pos_enc = nn.Identity()

        # Optionally project them to a target dimension.
        pos_enc = self._positions_projection(pos_enc)

        # Reshape inputs and positional encodings to allow add/concat.
        inputs = rearrange(
            inputs,
            "b t (n s) -> b (t n) s",
            n=self._num_patches_per_series,
            s=self._num_samples_per_patch,
        )
        pos_enc = repeat(pos_enc, "n s -> b (t n) s", b=batch_size, t=num_series)

        # Concat or add position encodings and inputs.
        if self._concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self._concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, None, inputs


class MetadataPreprocessor(perceiver.AbstractPreprocessor):
    def __init__(self, config: PerceiverConfig) -> None:
        self._config = config

    @property
    def num_channels(self) -> int:
        return self._config.d_model

    def forward(self, inputs: torch.Tensor) -> torch.tensor:
        """
        Args:
            inputs: tensor of size [batch, num_series, num_samples_per_series]
        """
        return inputs, None, inputs


class TimeSeriesDecoder:
    pass
