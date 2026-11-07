# Oracle wrapped-reference patch replacement

This optional experiment composes the corrupt render and pose-aligned wrapped
reference in pixel space before SD-Turbo's VAE encoder.

For every pixel-space patch:

1. require the configured fraction of geometrically valid wrapped pixels;
2. compare the render and wrapped-reference contexts against clean GT;
3. select the wrapped patch only when it satisfies the configured oracle
   margin and relative-gain requirements;
4. copy only geometrically valid wrapped pixels into the render, leaving holes
   transparent;
5. pass the resulting combined render to the unchanged DiFix model.

Clean GT is used only for the detached hard oracle decision. It is never copied
into the model input. The original render, reference, wrapped-reference view,
targets, attention routing, and loss definitions otherwise remain unchanged.

## Configuration

```text
--wrapped_reference_cache_dir PATH
--render_endpoint_mode wrapped_patch_override
--render_endpoint_override_policy oracle_gt
--render_endpoint_override_patch_size 16
--render_endpoint_override_valid_fraction 0.95
--render_endpoint_oracle_metric lpips
--render_endpoint_oracle_window_patches 3
--render_endpoint_oracle_margin 0.0
--render_endpoint_oracle_relative_gain 0.05
--render_endpoint_oracle_lpips_chunk_size 64
```

The LPIPS implementation is the exact PyIQA 0.1.16 LPIPS-VGG feature
calculation used by the PixelDiT oracle. Invalid wrapped pixels are filled from
the render before extracting the LPIPS context, exactly matching PixelDiT's
context construction. The replacement-validity threshold is independent from
the sparse-attention validity threshold.

`l1` and `mse` are also available as cheaper oracle metrics. `all_valid` can be
used instead of `oracle_gt` to copy every geometrically eligible patch.

## Logging

Training logs include:

```text
oracle_override_patch_fraction
oracle_eligible_patch_fraction
oracle_selected_of_eligible
```

Validation additionally logs `oracle_combined_*` image-quality metrics and
adds the selected-patch overlay and oracle-combined render to each image panel.
The original corrupt render remains the source for `input_*` metrics.
`restored_wrapped_*` remains the full-frame score, while
`restored_wrapped_valid_*` target-fills warp holes and reports coverage; its
PSNR/MAE/RMSE are normalized only over geometrically valid pixels.

The default `render_endpoint_mode=original` preserves the model's existing
two-view and three-view behavior. Validation now deliberately retains source
pixels in FP32 through metrics and oracle composition, so metrics may differ
slightly from older mixed-precision validation runs.

When building missing wrapped-reference cache entries online, the loader first
verifies that the required depth entry exists. Missing or incorrectly remapped
depth data therefore fails before an empty cache entry can be written.
