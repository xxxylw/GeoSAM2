# GeoSAM2 Product Context

This context defines the product language for turning 3D mesh inputs into part-labelled outputs. It exists to keep interactive and automatic segmentation requirements separate.

## Language

**Prompt-driven Part Segmentation**:
A segmentation mode where the user supplies one or more prompts that identify the target part or parts to extract from a mesh.
_Avoid_: Interactive segmentation, click segmentation, A mode

**Point Prompt**:
A user-supplied image-space point and label used to identify foreground or background for a target part.
_Avoid_: Click, marker

**Mask Prompt**:
A user-supplied image-space mask that identifies one or more target parts for prompt-driven segmentation.
_Avoid_: Segmentation image, mask file

**Automatic Part Segmentation**:
A segmentation mode where the system attempts to partition a mesh into parts without user prompts for each target part.
_Avoid_: Full segmentation, auto split, B mode

**Coarse Part Segmentation**:
An automatic segmentation target that prefers a small number of stable major parts over fine detail.
_Avoid_: Fine-grained split, exhaustive decomposition

**Reliable Coarse Part**:
A coarse part that is large enough, sufficiently covered by face labels, and not just an isolated fragment.
_Avoid_: Candidate mask, visual blob

**Geometry-driven Segmentation**:
Part segmentation that uses mesh geometry, rendered views, and visual masks without assigning semantic category names to parts.
_Avoid_: Semantic segmentation, category-based split

**Seed View Pair**:
Two rendered views used to generate automatic mask candidates from opposite sides of the source object.
_Avoid_: All views, random views

**Render Cache**:
A reusable set of rendered views and metadata for a source GLB and its render settings.
_Avoid_: Temporary output, intermediate folder

**Available GPU Set**:
The GPUs that a caller allows the segmentation system to choose from for GPU-capable work.
_Avoid_: Default CUDA device, hard-coded GPU

**Unsegmented Part**:
A single output part used when no reliable coarse subdivision is found for a source object.
_Avoid_: Failed part, missing segmentation

**Part-labelled Mesh**:
A mesh whose faces are assigned part labels so the segmented result can be inspected or reused downstream.
_Avoid_: Segmented result, output mesh

**Part-object GLB**:
A single GLB containing multiple part objects that can be independently shown or hidden in a mesh viewer while retaining the source asset's authored appearance where possible.
_Avoid_: Multiple output files, coloured mesh only

**Source Object Boundary**:
The object or mesh grouping already present in the input GLB before segmentation.
_Avoid_: Original layer, imported chunk

**Visualization Colours**:
Optional temporary colours assigned to part labels so a segmentation can be inspected visually; they are not the source mesh's authored PBR materials.
_Avoid_: New materials, recoloured asset

**Strict PBR Preservation**:
A requirement that segmented outputs retain the input GLB's authored PBR materials, UVs, texture references, and related appearance data.
_Avoid_: Best-effort material copy, preview colours

## Flagged Ambiguities

**Segmentation**:
This term is ambiguous unless qualified. Use **Prompt-driven Part Segmentation** when prompts define the target parts, and **Automatic Part Segmentation** when the system discovers parts without prompts.

**Coloured Output**:
This term is ambiguous because the source GLB can already contain authored PBR materials. Use **Visualization Colours** only when label colours are intentionally added for inspection.

**Successful Output**:
This term excludes preview-only exports. A successful default output must satisfy Strict PBR Preservation.

## Example Dialogue

Developer: "Should this GLB run through Prompt-driven Part Segmentation or Automatic Part Segmentation?"

Domain expert: "Run both. First let the user isolate a target part with prompts, then also offer an automatic pass that produces a Part-labelled Mesh without prompts. Preserve the original PBR materials unless Visualization Colours are explicitly requested."

Developer: "Which prompts should Prompt-driven Part Segmentation accept?"

Domain expert: "Support both Point Prompts and Mask Prompts. Mask Prompts are the direct path; Point Prompts can be converted into masks before 3D lifting."

Developer: "Should the default output be separate files per part?"

Domain expert: "No. The default should be a Part-object GLB so MeshLab and similar tools show independently toggleable parts in one file."

Developer: "Can the default output fall back to preview colours if materials are hard to preserve?"

Domain expert: "No. The default output must satisfy Strict PBR Preservation; preview colours are optional debug artifacts, not successful default output."

Developer: "If the source GLB already contains multiple objects, should segmentation merge across them?"

Domain expert: "No. Preserve each Source Object Boundary and split parts within each source object."

Developer: "How many parts should Automatic Part Segmentation produce?"

Domain expert: "Use Coarse Part Segmentation by default. Aim for two to five parts, but return an Unsegmented Part when no reliable coarse subdivision exists."

Developer: "How should automatic results be accepted?"

Domain expert: "Keep only Reliable Coarse Parts. A default run should target two to five parts, reject parts below about five percent of source-object area, cover about ninety-five percent of faces, and fall back to an Unsegmented Part when the subdivision is not reliable."

Developer: "Should automatic segmentation identify semantic categories?"

Domain expert: "No. Use Geometry-driven Segmentation by default; labels identify coarse parts, not names such as leg or handle."

Developer: "Should Automatic Part Segmentation generate candidates from every rendered view?"

Domain expert: "No. Start from a Seed View Pair so coarse parts can be discovered from opposite sides without encouraging over-fragmentation."

Developer: "Should rendered views be regenerated for every run?"

Domain expert: "No. Use a Render Cache keyed by the source GLB content and render settings so automatic and prompt-driven runs can reuse the same rendered views."

Developer: "Which GPUs may a segmentation run use?"

Domain expert: "Use one GPU per segmentation run, selected from the Available GPU Set. Default to the first visible GPU unless the caller restricts or selects another GPU."
