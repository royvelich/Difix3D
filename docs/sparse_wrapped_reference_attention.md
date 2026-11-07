# Sparse wrapped-reference attention

This branch adds an optional three-view training mode to the data-matched
SD-Turbo/DiFix fine-tuning code.

## Views and default policy

The view order is fixed:

1. corrupt render;
2. unwrapped rank-0 reference;
3. reference warped into the render pose.

The default sparse configuration is:

```text
query modes: all,all,valid_only

query \ source       render       reference       wrapped
render               all          all             valid_only
reference            all          all             valid_only
wrapped              all          all             valid_only
```

Thus every render/reference token issues a query. Only geometrically valid
wrapped tokens issue queries or provide keys/values. The routing is generic:
every query mode and every query/source edge accepts `all`, `valid_only`, or
`none`.

## Multiresolution validity

The dataset supplies a pixel validity mask `[B,1,H,W]`. The model pools it to
the actual VAE latent grid and then follows the U-Net's kernel-3, stride-2,
padding-1 downsampling geometry. For a 576x1024 image the attention masks are:

```text
72x128 -> 36x64 -> 18x32 -> 9x16
```

At each `attn1` block the implementation:

1. gathers permitted query tokens;
2. gathers and concatenates permitted key/value tokens;
3. runs scaled-dot-product attention without a quadratic policy mask;
4. scatters the attention updates into the dense view grids;
5. continues the original residual, text attention, feed-forward, and dense
   convolution paths.

This is attention sparsity, not strict end-to-end isolation. Invalid wrapped
regions still exist in the dense VAE/U-Net stream and can affect nearby
features through convolutions and normalization.

## Wrapped-reference cache

`PairedDataset` reads the PixelDiT precomputed `.npz` format:

```text
rgb:      uint8 [H,W,3]
validity: packbits(flattened Boolean mask, bitorder="little")
shape:    int32 [H,W]
```

Entries may use the PixelDiT path-hash layout
`<cache>/<key[:2]>/<key>.npz`. An explicit
`wrapped_reference_cache_key` manifest field is preferred when data paths have
been relocated. Record IDs and flat `<cache>/<key>.npz` files are also
supported.

## Training flags

```text
--wrapped_reference_cache_dir PATH
--validation_wrapped_reference_cache_dir PATH
--wrapped_reference_fill black|zero|corrupt
--sparse_mv_attention
--view_attention_query_modes all,all,valid_only
--view_attention_matrix \
  'all,all,valid_only;all,all,valid_only;all,all,valid_only'
--wrapped_reference_patch_valid_fraction 0.8
--wrapped_reference_denoise_mask all|valid_only
--wrapped_reference_loss_mask all|valid_pixels
```

The attention mask, scheduler-update mask, and supervision mask are separate
settings. Disabling the wrapped cache and sparse flag retains the official
two-view DiFix data and attention path.

## Verification

- Unit coverage compares sparse attention with an explicit gathered reference,
  including `all`, `valid_only`, `none`, and empty-validity cases.
- A real tiny Diffusers U-Net passes dense two-view and sparse three-view
  forward/backward tests with gradient checkpointing.
- A two-rank Gloo DDP smoke test uses unequal valid-token counts, including a
  rank with no valid wrapped tokens, and verifies exact parameter agreement.
- The multiview forward patch is installed only on the custom U-Net instances;
  importing it no longer monkey-patches every Diffusers transformer block in
  the process.
