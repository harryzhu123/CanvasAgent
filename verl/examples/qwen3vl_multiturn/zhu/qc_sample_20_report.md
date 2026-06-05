# QC Sample Report

Source dataset:
- `/nfsdata4/zhuhairui/verl/examples/qwen3vl_multiturn/zhu/qwen_stage12_full_2000_en_with_local_images.json`

Sampling method:
- Deterministic stratified sample of 20 records
- Coverage:
  - `generated_product_comparison_chain`: 3
  - `fine_grained_instance_trace_chain`: 3
  - `document_reverse_recover_translate_summary_chain`: 3
  - `weather_dual_surface_restore_chain`: 3
  - `poster_rewrite_reverify_summary_chain`: 2
  - `triple_image_sign_rebuild_transfer_chain`: 2
  - `multi_image_identity_cover_chain`: 2
  - `curved_label_recover_replace_reinsert_chain`: 2

Automatic checks:
- `20/20` sampled records are structurally valid
- `issues` is empty for all sampled records
- `image_prompt` count matches `input_spec.image_count` for all sampled records
- All sampled image paths exist
- No missing `required_tools` or `toolchainlength`

Manual review summary:
- Overall verdict: usable and mostly well-aligned
- Strong categories:
  - `curved_label_recover_replace_reinsert_chain`
  - `document_reverse_recover_translate_summary_chain`
  - `fine_grained_instance_trace_chain`
  - `weather_dual_surface_restore_chain`
  - `triple_image_sign_rebuild_transfer_chain`
- Acceptable but a bit synthetic:
  - `generated_product_comparison_chain`
  - `poster_rewrite_reverify_summary_chain`
- Main risk area:
  - `multi_image_identity_cover_chain`

Key findings:
1. Curved-label samples are strong.
The generated source images clearly contain curved, readable packaging labels, so the recover-replace-reinsert workflow is plausible.

2. Reverse-document samples are strong.
The sampled images visibly contain mirrored or tilted documents behind reflection/glass, which supports orientation correction and OCR-like recovery.

3. Fine-grained instance tracking looks usable.
The sampled shelf/table scenes contain multiple similar striped objects and distractors, so "pick the right one" is a meaningful requirement.

4. Weather-preservation samples are workable.
The sampled scene contains two distinct readable text surfaces, which is necessary for the "global weather edit + local text restoration" task.

5. Triple-image sign transfer is well-structured.
The three roles are clear:
- source sign with readable text
- blank carrier panel
- destination scene with a plausible mounting area

6. Prompt-only product-comparison tasks are valid but read slightly like specs rather than natural user requests.
They are still coherent, but their wording is a bit more "instruction blueprint" than conversational.

7. Poster/flyer rewrite samples are usable, but source text may be synthetic or non-English.
This is not fatal for the task, because the required structure is present, but it is something to watch if downstream OCR quality matters a lot.

8. The main quality risk is identity-cover generation.
In one reviewed case, the first portrait matched the intended event-portrait style well, but the second reference image drifted toward a generic studio full-body shot rather than a same-venue three-quarter portrait. The sample remains usable, but identity-consistency pressure is weaker than intended.

Representative sample IDs reviewed:
- `rl_batch5_0101`
- `rl_batch5_0201`
- `rl_batch5_0301`
- `rl_batch5_0202`
- `rl_batch5_0752`
- `rl_batch5_0751`
- `rl_batch5_0750`
- `rl_batch5_1551`
- `rl_batch5_1552`
- `rl_batch5_1553`

Recommendation:
- Safe to proceed with downstream usage
- If quality tightening is needed, prioritize improving `multi_image_identity_cover_chain` image prompts so the second portrait stays in the same event/context style as the first
