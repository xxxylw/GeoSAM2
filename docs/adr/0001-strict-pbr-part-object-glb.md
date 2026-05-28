# Strict PBR Part-object GLB Output

Default segmentation output must be a Part-object GLB that preserves the input GLB's authored PBR materials, UVs, texture references, and related appearance data. We will not treat GeoSAM2's current vertex-colour GLB export as a successful default output because it is useful for previewing labels but destroys the asset-fidelity guarantee needed for downstream mesh inspection and editing.

## Considered Options

- Use GeoSAM2's existing vertex-colour GLB export as the main output.
- Export separate GLB files per part.
- Export a single Part-object GLB with Strict PBR Preservation.

## Consequences

The GeoSAM2 inference path may still produce face labels and optional preview artifacts, but final export requires a glTF-aware splitting step that can fail if PBR preservation cannot be guaranteed.
