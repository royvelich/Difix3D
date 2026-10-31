# Difix Architecture and Wrapped-Reference Attention Design

Date: 2026-10-31

## Purpose

This document records our discussion and the independent Claude review of:

- how reference-conditioned Difix processes images;
- where cross-view attention occurs;
- the shapes of the U-Net feature maps and tokens;
- how Difix tokens relate to pixel-space regions;
- why the PixelDiT validity-mask implementation is simpler;
- how to add a wrapped-reference view and valid-region attention to Difix.

The branch containing this document was created directly from the latest official
`nv-tlabs/Difix3D` commit at the time of the review:
`c76edc595586e16732c91ddee82f3a6d83a8a9cc`.

## 1. Official reference-conditioned Difix

For a batch size `B`, Difix starts with two image views:

```text
corrupt render:  [B, 3, 576, 1024]
reference image: [B, 3, 576, 1024]
```

They are stacked as:

```text
[B, 2, 3, 576, 1024]
```

and flattened into the batch dimension before the VAE:

```text
[2B, 3, 576, 1024]
```

### VAE encoding

SD-Turbo's VAE downsamples the image by a factor of eight:

```text
[2B, 3, 576, 1024]
        |
        v
[2B, 4, 72, 128]
```

The prompt embedding is repeated for both views:

```text
original prompt embedding: [B, 77, 1024]
repeated prompt embedding: [2B, 77, 1024]
```

### U-Net processing

The cross-view interaction happens repeatedly **inside** the SD-Turbo U-Net.
There is no separate attention stage after the U-Net.

A simplified U-Net stage is:

```python
features = convolution_or_resnet(features)
features = joint_view_attention(features)
features = downsample_or_upsample_if_needed(features)
```

Ordinary convolution/ResNet operations generally preserve `H x W`, although
they can change the channel dimension. Dedicated downsampling and upsampling
operations change `H x W`.

For example:

```text
initial latent:      [2B,    4, 72, 128]
initial convolution: [2B,  320, 72, 128]
downsample:          [2B,  320, 36,  64]
next stage:          [2B,  640, 36,  64]
```

### Joint image-token attention

At an attention block, a feature map is flattened spatially:

```text
feature map: [2B, C, h, w]
tokens:      [2B, h*w, C]
```

The render and reference tokens are then merged along the token dimension:

```text
render tokens:    [B, h*w, C]
reference tokens: [B, h*w, C]

joint tokens:     [B, 2*h*w, C]
```

The U-Net's first attention operation, `attn1`, performs self-attention over the
joint sequence. It is then split back into render and reference rows:

```python
joint = concatenate(render_tokens, reference_tokens)
joint = self_attention(joint)
render_tokens, reference_tokens = split(joint)
```

The text cross-attention, feed-forward network, convolutions, and normalization
remain per-view. Only `attn1` is changed into joint image-view attention.

Official Difix uses unrestricted, symmetric attention:

```text
render queries    -> render + reference keys
reference queries -> render + reference keys
```

It does not use depth, correspondence, camera geometry, or a reference-validity
mask. The model must learn useful cross-view matches from feature content.

### U-Net attention shapes

For 576 x 1024 inputs:

| U-Net location | Spatial grid | Tokens per view | Joint tokens for two views | Token feature dimension |
|---|---:|---:|---:|---:|
| Down 1 / Up 3 | 72 x 128 | 9,216 | 18,432 | 320 |
| Down 2 / Up 2 | 36 x 64 | 2,304 | 4,608 | 640 |
| Down 3 / Up 1 | 18 x 32 | 576 | 1,152 | 1,280 |
| Bottleneck | 9 x 16 | 144 | 288 | 1,280 |

For example, at 36 x 64:

```text
render:    [B, 2304, 640]
reference: [B, 2304, 640]
joint:     [B, 4608, 640]
```

Here, `640` is the feature dimension of each token. The views are concatenated
along the `2304` token dimension, not along the feature dimension.

### One-step restoration and decoding

The U-Net predicts a latent update at fixed timestep 199:

```text
U-Net prediction: [2B, 4, 72, 128]
```

A one-step scheduler update produces restored latents with the same shape:

```text
restored latents: [2B, 4, 72, 128]
```

The VAE decodes both rows:

```text
[2B, 4, 72, 128]
        |
        v
[2B, 3, 576, 1024]
        |
        v
[B, 2, 3, 576, 1024]
```

The two training targets are:

```text
restored render    -> clean render-pose ground truth
restored reference -> original reference image
```

Normal inference returns the restored render and discards the decoded reference
output.

## 2. Token grid location versus receptive field

A Difix U-Net token is not inherently a fixed 16 x 16 pixel patch.

There are two different concepts:

1. **Nominal grid footprint or stride:** the pixel-space bin associated with the
   token's grid coordinate.
2. **Receptive field:** all input pixels that can influence the token's feature
   vector.

The nominal grid mapping is:

| Attention grid | Nominal pixel stride |
|---:|---:|
| 72 x 128 | 8 x 8 pixels |
| 36 x 64 | 16 x 16 pixels |
| 18 x 32 | 32 x 32 pixels |
| 9 x 16 | 64 x 64 pixels |

Only the 36 x 64 U-Net level has the same nominal grid as PixelDiT's 16 x 16
patch grid.

For an arbitrary image size `(H, W)` and attention grid `(h, w)`, token `(y, x)`
is anchored to the nominal pixel region:

```text
rows:    [y * H / h, (y + 1) * H / h)
columns: [x * W / w, (x + 1) * W / w)
```

This does not mean that the token exclusively represents those pixels. VAE and
U-Net convolutions have overlapping receptive fields. Group normalization also
aggregates statistics across spatial positions. After self-attention, a token
can depend on every token that its attention policy allows.

Therefore, a token-validity mask should mean:

> The token is spatially anchored in a sufficiently valid wrapped-reference
> region.

It cannot guarantee:

> The token's feature vector contains information exclusively from that exact
> pixel rectangle.

## 3. Proposed wrapped-reference extension

The extended model has three views:

```text
R = corrupt render
U = original unwrapped reference
W = reference warped into the render pose
```

Input shape:

```text
[B, 3, 3, 576, 1024]
```

After flattening views and VAE encoding:

```text
[3B, 4, 72, 128]
```

The wrapped-reference mode should be optional. The same code must retain a
two-view official-Difix mode for the apples-to-apples baseline.

### Recommended directional attention policy

The initial recommended policy is:

| Query view | Allowed key/value views |
|---|---|
| Render | Render + unwrapped reference + valid wrapped reference |
| Unwrapped reference | Unwrapped reference only |
| Wrapped reference | Unwrapped reference + valid wrapped reference |

The unwrapped-reference policy can be configurable. A symmetric mode may retain
official Difix behavior:

```text
unwrapped-reference queries -> render + unwrapped-reference keys
```

Keeping the unwrapped reference isolated is cleaner for the extended mode. It
prevents an indirect information path across consecutive layers:

```text
render -> unwrapped reference -> render
```

Suggested selectable policies:

- `official_joint`: all official views jointly attend as in released Difix;
- `render_all`: render sees all views, reference stays isolated, and wrapped
  sees reference plus itself;
- `joint_three_view`: all three views attend all three views;
- additional policies can be represented with the same query-view/key-view
  routing table.

### Wrapped-reference restoration target

The proposed outputs and targets are:

| Output row | Target |
|---|---|
| Render | Clean render-pose ground truth |
| Unwrapped reference | Original unwrapped reference |
| Wrapped reference | Clean render-pose ground truth |

The wrapped-reference loss scope should be configurable:

- `valid`: supervise only valid wrapped-reference pixels;
- `full`: supervise the complete clean render-pose target.

The invalid input region should use a configurable neutral fill, with black as
the intended initial setting. Attention masking alone does not prevent invalid
input values from influencing nearby features through VAE/U-Net convolutions or
normalization.

## 4. Multiresolution wrapped-validity masks

The wrapped-reference pixel validity and confidence should be aggregated
directly into every U-Net attention grid:

```text
pixel validity/confidence
        |
        +-> 72 x 128 token mask
        +-> 36 x 64 token mask
        +-> 18 x 32 token mask
        `->  9 x 16 token mask
```

It is more accurate to derive every mask from the original pixel-level data than
to derive all masks from the existing 36 x 64 PixelDiT patch mask.

Conceptually:

```python
valid_fraction = area_pool(pixel_validity, output_size=(h, w))

valid_weight = area_pool(
    pixel_validity * pixel_confidence,
    output_size=(h, w),
)

valid_count = area_pool(
    pixel_validity,
    output_size=(h, w),
)

mean_confidence = valid_weight / clamp(valid_count, epsilon)

token_valid = (
    (valid_fraction >= valid_fraction_threshold)
    & (mean_confidence >= confidence_threshold)
)
```

The exact `(h, w)` must be carried from the spatial feature map into the
attention implementation before it is flattened. Inferring `(h, w)` only from
`N = h*w` is unsafe for rectangular or odd-sized inputs.

For image dimensions not divisible by every U-Net scale, pooling must follow the
same padding/rounding behavior as the U-Net. Our 576 x 1024 working resolution
is exactly divisible by 64.

Possible conservative options include:

- requiring high valid coverage for a token;
- eroding validity at deeper U-Net levels, where receptive fields are larger;
- using a soft confidence bias rather than only a hard Boolean threshold.

The first implementation should keep the hard-mask behavior simple and
diagnosable. Soft weighting can be evaluated separately.

## 5. Why a normal Diffusers mask is insufficient

The released custom U-Net accepts a normal key-padding mask with conceptual
shape:

```text
[B, key_tokens]
```

It becomes a broadcastable key bias:

```text
[B, 1, key_tokens]
```

This mask can say that a key is globally usable or unusable. It cannot express:

```text
render queries may use wrapped keys
reference queries may not use wrapped keys
```

It also cannot use one fixed key mask for U-Net layers whose token counts change
between 9,216, 2,304, 576, and 144 tokens per view.

A full asymmetric mask with shape `[B, Q, K]` could represent the policy, but it
would be unnecessarily large and could interfere with efficient attention
kernels.

The recommended implementation is a custom `attn1` path that splits queries by
view and performs one attention operation per query view:

```python
outputs = []

for query_view in views:
    source_views = policy.allowed_sources(query_view)

    q = queries[query_view]
    k = concatenate(keys[source] for source in source_views)
    v = concatenate(values[source] for source in source_views)

    key_valid = concatenate(
        validity[source] for source in source_views
    )

    output = scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=key_valid[:, None, None, :],
    )

    output *= query_validity[query_view]
    outputs.append(output)

return concatenate(outputs)
```

This provides directional view policies and wrapped-token validity without
materializing a quadratic query-by-key mask.

Every query row must retain at least one valid source set, normally its own view,
so that no attention row is completely masked.

## 6. Why PixelDiT did not require multiresolution handling

PixelDiT already implements the same split-query strategy explicitly in
`apply_view_attention_policy`.

For every query view, it:

1. selects the permitted source views;
2. slices the query tokens for that view;
3. concatenates only the permitted key/value streams;
4. constructs a broadcast key-validity mask;
5. calls PyTorch scaled-dot-product attention;
6. masks invalid query outputs.

It does not construct a full `[Q, K]` policy mask.

PixelDiT is simpler because its 16 x 16 patch grid remains fixed across all 14
DiT blocks:

```text
576 x 1024 image
        |
        v
36 x 64 patch grid
        |
        v
2,304 patch tokens per view in every DiT block
```

Its wrapped validity can therefore be pooled once:

```python
patch_fraction = average_pool(
    pixel_validity,
    kernel_size=patch_size,
    stride=patch_size,
)
```

The resulting `[B, views, 2304]` mask is reused in every DiT block.

Difix requires the same basic query-splitting mechanism but needs separate masks
for all of its U-Net resolutions.

## 7. Implementation hazards identified in review

The following issues should be handled during implementation:

1. `num_views` is hardcoded to two in the released multiview attention code.
2. The current global monkey-patch of `BasicTransformerBlock.forward` affects
   other Diffusers models imported into the same process.
3. The attention block currently receives flattened tokens without the original
   spatial `(h, w)`.
4. Standard Diffusers key-padding masks cannot represent directional view
   policies.
5. Invalid wrapped pixels can still influence features through convolutions and
   normalization before an attention mask is applied.
6. The released Diffusers inference pipeline concatenates batched images and
   references in a layout that is compatible with `B=1` but does not match the
   interleaved `(batch, view)` layout expected by the multiview reshape for
   general `B>1`.
7. There are no explicit view-ID embeddings in released Difix.
8. Three views increase attention computation and require a GPU memory smoke
   test before full training.

## 8. Baseline and extension must remain distinct

The implementation should support two clearly reported experiment classes:

### Official two-view Difix baseline

```text
views: corrupt render + unwrapped reference
attention: released symmetric joint attention
targets: clean render + identity reference
```

This mode is the data-matched Difix baseline.

### Three-view wrapped-reference extension

```text
views: corrupt render + unwrapped reference + wrapped reference
attention: configurable directional policy with wrapped validity masks
targets: clean render + identity reference + clean render-pose target
```

This is not the official Difix baseline. It is a separate ablation combining
Difix with our geometry-derived wrapped-reference mechanism.

## 9. Verification status

Claude independently reviewed the released Difix U-Net, model, and pipeline
paths. The review agreed that:

- cross-view mixing is token-based and occurs in `attn1` inside the U-Net;
- the 8/16/32/64-pixel nominal stride table is correct for 576 x 1024;
- token stride must not be confused with the token's receptive field;
- each U-Net resolution needs its own validity mask;
- true `(h, w)` must be preserved before flattening;
- a custom split-query attention implementation is required for asymmetric
  view policies.

Claude additionally recommended conservative masks at deeper levels, neutral
invalid-region filling, and explicit tests for indirect cross-view leakage.

No wrapped-reference Difix implementation had been made when these notes were
written.
