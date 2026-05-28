# PRD: GLB Part-object Segmentation

## Problem Statement

Users have GLB assets with authored PBR materials and need a way to split them into inspectable parts without destroying asset fidelity. The desired output is a single GLB that opens in MeshLab or similar mesh viewers as multiple independently visible objects, while preserving the original PBR materials, UVs, texture references, and related appearance data.

GeoSAM2 already produces per-face 3D part labels from multi-view segmentation, but its current exported GLB is a vertex-colour preview. That preview is useful for inspection but is not an acceptable default output for this workflow because it does not satisfy Strict PBR Preservation.

## Solution

Build a Python API and CLI that take an input GLB and produce a Strict PBR-preserving Part-object GLB. The default mode is Automatic Part Segmentation using Geometry-driven Segmentation, Coarse Part Segmentation, a Seed View Pair, and reliability rules that prefer two to five major parts. Prompt-driven Part Segmentation is also supported through Mask Prompts and Point Prompts.

The pipeline reuses GeoSAM2 for rendering, 2D mask propagation, and 2D-to-3D face labelling. A new glTF-aware export stage uses those face labels to split the original GLB into independently toggleable part objects while preserving Source Object Boundaries and PBR data. Optional preview outputs may use Visualization Colours, but preview artifacts never count as Successful Output.

## User Stories

1. As a 3D asset user, I want to input a GLB and receive a Part-object GLB, so that I can inspect major parts in MeshLab.
2. As a 3D asset user, I want automatic segmentation to be the default mode, so that I do not need to provide prompts for common assets.
3. As a 3D asset user, I want the output parts to be independently hideable and visible, so that I can inspect each part in isolation.
4. As a 3D asset user, I want the output to preserve PBR materials, so that the segmented asset remains visually faithful to the source asset.
5. As a 3D asset user, I want the system to fail when Strict PBR Preservation cannot be guaranteed, so that I do not accidentally consume degraded assets.
6. As a 3D asset user, I want optional Visualization Colours, so that I can quickly inspect labels without replacing the default PBR-preserving output.
7. As a 3D asset user, I want Automatic Part Segmentation to produce coarse parts, so that the result is manageable rather than over-fragmented.
8. As a 3D asset user, I want automatic output to target two to five major parts, so that MeshLab layers remain easy to navigate.
9. As a 3D asset user, I want unstructured assets to become an Unsegmented Part, so that the system does not invent unreliable splits.
10. As a 3D asset user, I want Source Object Boundaries preserved, so that existing authoring structure is not lost.
11. As a 3D asset user, I want multi-object GLBs to be split within each source object, so that unrelated source objects are not merged together.
12. As a 3D asset user, I want Prompt-driven Part Segmentation, so that I can isolate a specific target part when automatic segmentation is not enough.
13. As a 3D asset user, I want Mask Prompts supported, so that I can provide an existing 2D mask directly.
14. As a 3D asset user, I want Point Prompts supported, so that I can mark a part with image-space points and let GeoSAM2 produce the mask.
15. As a command-line user, I want a CLI entry point, so that I can run segmentation from scripts and terminals.
16. As a Python developer, I want a Python API, so that I can integrate segmentation into services, UI tools, or batch pipelines.
17. As a repeat user, I want a Render Cache, so that repeated runs on the same GLB do not rerender all views.
18. As a repeat user, I want cache invalidation based on GLB content and render settings, so that stale render data is not reused incorrectly.
19. As a GPU workstation user, I want the run to use one selected GPU, so that GPU usage is predictable.
20. As a GPU workstation user, I want to choose the GPU, so that I can avoid contention with other work.
21. As a downstream tool developer, I want face labels saved alongside the GLB, so that I can consume labels programmatically.
22. As a downstream tool developer, I want a part manifest, so that I can map part objects back to source objects, labels, face counts, and area ratios.
23. As a debugging user, I want render cache keys and warnings saved, so that I can understand how a result was produced.
24. As a debugging user, I want optional frame and colored-preview artifacts, so that I can inspect segmentation failures.
25. As an implementer, I want a clear separation between label inference and PBR-preserving export, so that the preview path does not leak into Successful Output.

## Implementation Decisions

- Implement a deep Python API centered on a stable `segment_glb` interface. It accepts input path, output directory or output path, mode, prompt inputs, cache directory, GPU selection, strict PBR settings, and preview options.
- Implement the CLI as a thin wrapper around the Python API. The CLI is the first acceptance entry point, but core behavior must remain importable.
- Support two first-version modes: Automatic Part Segmentation and Prompt-driven Part Segmentation. Refinement of an existing automatic result is out of scope for the first version.
- Make Automatic Part Segmentation the default mode.
- Use Geometry-driven Segmentation only. Do not add semantic category naming or VLM/LLM-assisted part classification.
- Use Coarse Part Segmentation for automatic mode. Target two to five Reliable Coarse Parts.
- Generate automatic candidates from a Seed View Pair rather than every rendered view.
- Treat candidates below about five percent of source-object area as too small for default coarse output.
- Require about ninety-five percent face-label coverage for a reliable segmented result.
- Return an Unsegmented Part when no reliable coarse subdivision exists.
- Preserve Source Object Boundaries. If the input GLB already has multiple source objects, split parts within each source object rather than merging labels globally.
- Preserve Strict PBR Preservation for Successful Output. This follows ADR-0001 and rejects the current GeoSAM2 vertex-colour GLB export as the default output path.
- Add a glTF-aware exporter module that takes the original GLB and face labels and produces a single Part-object GLB.
- Keep GeoSAM2's existing vertex-colour output available only as an optional preview artifact.
- Add a Render Cache keyed by input GLB content hash, render settings hash, and render script version or hash.
- Use one GPU per segmentation run. Default to the first visible GPU and allow explicit GPU selection.
- Define output structure with the Part-object GLB at the top level, labels in a labels area, optional previews in a preview area, and debug metadata in a debug area.
- Include a part manifest that records run status, part object names, source object names, labels, face counts, and area ratios.
- Preserve debug artifacts when strict PBR export fails, but do not write the final successful Part-object GLB.

## Testing Decisions

- Tests should assert external behavior: file layout, manifest contents, cache reuse, GPU selection behavior, strict PBR failure behavior, source-object preservation, and output object counts.
- Avoid tests that assert private implementation details of GeoSAM2 internals.
- Unit test the Render Cache key generation with same-name different-content GLBs, changed render settings, and changed render-script hash.
- Unit test the CLI argument layer as a thin wrapper around the Python API contract.
- Unit test part filtering rules for Reliable Coarse Parts, including area threshold, face coverage threshold, target part count, and Unsegmented Part fallback.
- Unit test manifest generation from representative part-label data.
- Integration test Prompt-driven Part Segmentation with a Mask Prompt using existing rendered example data.
- Integration test Point Prompt flow by generating a mask and passing it through the same 3D lifting path.
- Integration test Strict PBR-preserving export with a small GLB fixture containing PBR material, UVs, and texture references.
- Integration test Source Object Boundary behavior with a small multi-object GLB fixture.
- Smoke test the full CLI on a CUDA machine with one single-object multi-part GLB, one multi-object GLB, and one unstructured GLB.
- Existing prior art is script-level smoke testing through the current example flow and compile checks, since the repo does not currently have a formal test suite.

## Out of Scope

- Training, fine-tuning, dataset creation, and evaluation pipelines.
- Semantic part naming such as leg, seat, handle, or wheel.
- VLM, LLM, or category-based segmentation assistance.
- Multi-GPU execution for a single segmentation run.
- Batch scheduling across multiple GPUs.
- Refinement mode for editing an existing automatic result.
- Web UI or interactive MeshLab integration.
- Guaranteeing a split when no Reliable Coarse Parts are found.
- Treating preview-only colored outputs as successful default outputs.

## Further Notes

The main technical risk is the PBR-preserving Part-object GLB exporter. GeoSAM2 can provide face labels, but Successful Output depends on splitting the original glTF structure without losing materials, UVs, textures, normals, transparency behavior, or source object boundaries. This risk is captured in ADR-0001 and should be validated early with small GLB fixtures before polishing the CLI.
