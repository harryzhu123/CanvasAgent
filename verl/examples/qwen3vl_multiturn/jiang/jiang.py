#!/usr/bin/env python3
"""Generate a third ultra-hard RL task batch with longer chains and more cross-image work."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


OUTPUT_JSONL = "rl_batch_1000_v3_ultra.jsonl"
OUTPUT_MD = "rl_batch_1000_v3_ultra_preview.md"
RNG_SEED = 20260407 + 33

DIFFICULTY_TARGETS = {
    "expert": 500,
    "extreme": 350,
    "ultra": 150,
}

GLOBAL_QUALIFIERS = [
    "Treat untouched regions as part of the evaluation target, not just the edited area.",
    "Do not use a shortcut that sacrifices local correctness for global smoothness.",
    "Keep geometry, material cues, and readability jointly consistent.",
    "Assume a reviewer will zoom into both edges and text-bearing surfaces.",
    "Preserve object identity through every extraction and transfer step.",
    "Do not let enhancement steps rewrite structure that earlier steps established.",
    "Use conservative local corrections when multiple interpretations are possible.",
    "Keep scale, perspective, and layer order believable at every stage.",
]

DELIVERY_NOTES = [
    "Only terminate when each sub-goal has visual evidence of being satisfied.",
    "Do not infer hidden content that is not supported by the image.",
    "Later polish steps must not overwrite earlier localized fixes.",
    "If a step depends on orientation or localization, resolve that dependency first.",
    "Assume the final check will compare edited targets against untouched context.",
]

TOOL_INFO = {
    "ImageGeneration": "generation",
    "ImageEdit": "editing",
    "ImageCrop": "cropping",
    "ImageRotate": "rotation",
    "ImageFlip": "flipping",
    "ImageGrounding": "grounding",
    "ImageSAM": "masking",
    "ImageExtract": "extraction",
    "ImageSR": "super-resolution",
    "Overlayer": "overlay",
    "OCR": "ocr",
}

TASK_BLUEPRINTS = [
    {
        "task_type": "triple_image_signage_transfer_pipeline",
        "difficulty": "ultra",
        "cross_image": True,
        "required_tools": ["OCR", "ImageGrounding", "ImageCrop", "ImageSR", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "ImageSR"],
        "preferred_tool_order": ["OCR", "ImageGrounding", "ImageCrop", "ImageSR", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "ImageSR"],
        "scene_templates": [
            "source storefront photo, clean facade plate, and rainy street mockup",
            "mall sign photo, blank acrylic panel, and transit station poster wall",
            "bookshop window photo, neutral signboard panel, and nighttime avenue scene",
            "cafe facade photo, replacement lightbox plate, and airport corridor ad frame",
        ],
        "prompt_templates": [
            "Use the {scene}: recover the readable sign text from the source, isolate the sign region cleanly, rebuild that text into {language} on the clean panel, then place the finished sign into the destination scene while preserving reflection logic and sharpening only at the end.",
            "Across the {scene}, extract the real sign content from the first image, enhance weak text before reading it, rebuild the sign in {language} on the second image, then composite that rebuilt sign into the third image without breaking perspective, reflections, or final realism.",
            "Work across the {scene}: localize and recover the physical signage from the source view, translate it into {language}, reconstruct it on the blank sign panel, and transfer that result into the destination street scene with only a late clarity pass.",
        ],
        "constraints": [
            "Text recovery must happen before sign reconstruction.",
            "Sign isolation must be local and should not drag surrounding facade textures into the new panel.",
            "Final composite must preserve reflection logic in the destination scene.",
            "The final sharpen pass should happen only after placement is complete.",
        ],
        "success_criteria": [
            "Readable source sign text is recovered correctly.",
            "Rebuilt translated sign is confined to the clean panel.",
            "Destination composite matches sign perspective and layer order.",
            "Reflections and highlights remain believable in the final scene.",
        ],
        "failure_traps": [
            "Translating reflected or mirrored text instead of the real sign.",
            "Sharpening too early and amplifying noise before extraction.",
            "Compositing the rebuilt sign before local cleanup is finished.",
        ],
        "hard_negative_actions": [
            "Applying global ImageEdit to the destination scene before sign reconstruction exists.",
            "Skipping local extraction and trying to repaint the sign directly into the final scene.",
            "Running the final ImageSR before Overlayer.",
        ],
        "reward_focus": ["multi-image planning", "text recovery", "local extraction", "composite realism", "late-stage enhancement"],
        "why_hard": "This is a true three-image chain with recovery, reconstruction, transfer, and late validation constraints that are highly order-sensitive.",
        "variables": {
            "language": ["Chinese", "Japanese", "Arabic", "Spanish", "French"],
        },
    },
    {
        "task_type": "multi_image_subject_cleanup_storyboard_chain",
        "difficulty": "ultra",
        "cross_image": True,
        "required_tools": ["ImageGrounding", "ImageCrop", "ImageSR", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "OCR", "Overlayer"],
        "preferred_tool_order": ["ImageGrounding", "ImageCrop", "ImageSR", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "OCR", "Overlayer"],
        "scene_templates": [
            "four noisy event photos of the same athlete plus a clean title card",
            "four travel portraits of the same guide plus a blank story cover",
            "four backstage musician photos plus a simple promo card",
            "four casual speaker photos plus a presentation opener slide",
        ],
        "prompt_templates": [
            "From the {scene}, isolate the same person consistently across the photo set, recover weak details where needed, build a clean hero cutout, place it onto the cover card, extract any readable name text from the original set, and add that text as a clean title on the final card.",
            "Use the {scene} to create a consistent cover image: select the best subject geometry across the source photos, normalize detail and identity, extract the final hero subject, composite it onto the blank cover, then recover and overlay the subject name if visible.",
            "Across the {scene}, stabilize subject identity from the source images, create one clean extracted hero figure, transfer that figure to the title card, then OCR any readable name information from the original photos and overlay it as a neat final heading.",
        ],
        "constraints": [
            "Identity drift across source photos is unacceptable.",
            "Text overlay should happen after the hero composition is stabilized.",
            "Subject extraction must preserve hairline and shoulder edges.",
            "If readable name text exists, it must be taken from the image evidence rather than invented.",
        ],
        "success_criteria": [
            "One stable hero subject is produced from the photo set.",
            "Hero cutout is cleanly composited onto the cover card.",
            "Visible subject naming is recovered only from image evidence.",
            "Final cover looks coherent rather than stitched from multiple sources.",
        ],
        "failure_traps": [
            "Mixing incompatible face angles into one unstable subject.",
            "Overlaying guessed title text with no visual basis.",
            "Adding title before the final subject placement is stable.",
        ],
        "hard_negative_actions": [
            "Trying global editing across all photos instead of isolating the subject.",
            "Running OCR before deciding which source image best supports readable text.",
            "Using Overlayer on the cover before extraction quality is acceptable.",
        ],
        "reward_focus": ["identity consistency", "edge quality", "evidence-grounded text", "cover composition"],
        "why_hard": "The agent must solve multi-image subject consistency, clean extraction, destination composition, and evidence-grounded titling in one chain.",
        "variables": {},
    },
    {
        "task_type": "recover_replace_curve_reinsert_verify_chain",
        "difficulty": "extreme",
        "cross_image": False,
        "required_tools": ["ImageRotate", "ImageGrounding", "ImageCrop", "ImageSR", "OCR", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer"],
        "preferred_tool_order": ["ImageRotate", "ImageGrounding", "ImageCrop", "ImageSR", "OCR", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer"],
        "scene_templates": [
            "helmet sticker photo with curved lettering",
            "arched banner on a sports stand",
            "plastic drink cup sleeve with warped text",
            "curved cosmetic bottle label",
            "paper bowl rim graphic with bent wording",
        ],
        "prompt_templates": [
            "In the {scene}, correct the reading angle only as much as necessary, isolate the text-bearing region, recover the wording, extract that local surface, replace the wording with '{replacement_text}', and reinsert it so the curvature still looks physically attached.",
            "For the {scene}, recover distorted text through minimal geometric correction, crop and enhance the local text area, read it, extract the actual surface patch, replace the wording with '{replacement_text}', and place the edited patch back without flattening the surface.",
            "Use the {scene} to run a full recover-and-reinsert cycle: make the text readable, extract the correct curved patch, rewrite it to '{replacement_text}', and restore it to the same surface with believable material continuity.",
        ],
        "constraints": [
            "Recovery steps should not permanently flatten the final surface.",
            "Extraction must correspond to the true text-bearing patch, not a surrounding decorative area.",
            "Replacement wording must inherit the original curvature and local texture logic.",
            "Reinsertion should preserve edge continuity around the patch.",
        ],
        "success_criteria": [
            "Original wording becomes readable enough to guide replacement.",
            "Correct local patch is extracted and edited.",
            "Final text follows curved geometry naturally.",
            "Patch boundaries are visually integrated after reinsertion.",
        ],
        "failure_traps": [
            "Editing text directly on the raw warped region without recovery.",
            "Extracting too large a patch and damaging surrounding graphics.",
            "Returning a flat-looking text block to a curved surface.",
        ],
        "hard_negative_actions": [
            "Using ImageEdit before text recovery and patch isolation.",
            "Skipping extraction and trying to overlay flat text back onto the surface.",
            "Treating the rotated recovery view as the final output image.",
        ],
        "reward_focus": ["minimal correction", "text recovery", "patch extraction", "surface reintegration"],
        "why_hard": "This is not just text replacement; it is a recover-extract-edit-reinsert cycle where each step preserves geometry for the next.",
        "variables": {
            "replacement_text": ["CITY PASS", "NIGHT RIDE", "LIMITED BLEND", "SUMMER SET", "ROUTE 24"],
        },
    },
    {
        "task_type": "generate_then_build_dual_panel_comparison_chain",
        "difficulty": "expert",
        "cross_image": False,
        "required_tools": ["ImageGeneration", "ImageGrounding", "ImageCrop", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "OCR"],
        "preferred_tool_order": ["ImageGeneration", "ImageGrounding", "ImageCrop", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "OCR"],
        "scene_templates": [
            "product ad with one hero bottle and accessory objects",
            "sports poster with one athlete and repeated props",
            "food ad with one main bowl and multiple side dishes",
            "travel card layout with a hero sign and secondary icons",
        ],
        "prompt_templates": [
            "Generate a {scene}, isolate the main target object, create an edited variant with {attribute_value} {attribute}, place the original and edited versions into a side-by-side comparison panel, then OCR any major generated headline and include it as a small comparison caption.",
            "Create a {scene}, extract the hero object, produce a locally edited version with {attribute_value} {attribute}, build a clean before-versus-after panel from both objects, and recover the main generated headline as a compact caption.",
            "Make a {scene}; after generation, separate the hero object, edit it locally to add {attribute_value} {attribute}, assemble original and edited versions in a comparison strip, then read the main headline and add it as a neat caption line.",
        ],
        "constraints": [
            "The edited variant should remain recognizably the same object.",
            "Comparison panel must clearly separate original and edited states.",
            "Caption text should come from OCR on the generated image rather than being invented.",
            "The original generated scene should remain intact outside the panel area.",
        ],
        "success_criteria": [
            "Hero object is correctly isolated from the generated scene.",
            "Edited variant changes only the intended local attribute.",
            "Comparison panel cleanly presents original versus edited object.",
            "Caption is grounded in visible generated text.",
        ],
        "failure_traps": [
            "Editing the whole scene instead of the hero object.",
            "Losing object identity between original and edited variants.",
            "Guessing a caption rather than OCR-ing visible text.",
        ],
        "hard_negative_actions": [
            "Running OCR before the scene and panel layout stabilize.",
            "Using ImageEdit before a clean hero object is isolated.",
            "Placing the panel before both object states exist.",
        ],
        "reward_focus": ["generation reliability", "object identity", "comparison clarity", "evidence-grounded caption"],
        "why_hard": "The task combines generation, local extraction, controlled object editing, structured comparison layout, and OCR-grounded captioning.",
        "variables": {
            "attribute": ["trim", "label accent", "surface finish", "stripe pattern", "badge color"],
            "attribute_value": ["copper", "forest green", "matte black", "warm orange", "deep blue"],
        },
    },
    {
        "task_type": "document_reverse_recover_translate_relayout_chain",
        "difficulty": "extreme",
        "cross_image": False,
        "required_tools": ["ImageFlip", "ImageRotate", "ImageCrop", "ImageSR", "OCR", "ImageEdit", "Overlayer", "ImageSR"],
        "preferred_tool_order": ["ImageFlip", "ImageRotate", "ImageCrop", "ImageSR", "OCR", "ImageEdit", "Overlayer", "ImageSR"],
        "scene_templates": [
            "mirrored event notice photographed from the side",
            "reversed menu board taken through glass",
            "crooked bilingual instruction card in a selfie shot",
            "tilted transit notice reflected in a window",
        ],
        "prompt_templates": [
            "Take the {scene}, correct the mirrored and angled reading direction, isolate the important text region, enhance it, recover the text, translate the main message into {language}, place the translated version as a clean summary block beside the original, and only then sharpen the final result.",
            "For the {scene}, undo reversal and tilt first, crop to the useful text, improve clarity, OCR the content, translate the main message into {language}, add that translation as a side summary panel, and keep sharpening as the final step only.",
            "Recover the {scene} into readable form, extract the main message, translate it into {language}, relayout the translation in a clean side block next to the source document, and apply only late-stage enhancement.",
        ],
        "constraints": [
            "Reading direction fixes must precede OCR.",
            "Translation panel should not occlude the original source evidence.",
            "Main message translation should summarize the source, not rewrite unrelated text.",
            "The last sharpening pass must not damage already-corrected text edges.",
        ],
        "success_criteria": [
            "Reversal and tilt are corrected before OCR.",
            "Main text becomes readable enough to transcribe.",
            "Translated summary panel is cleanly placed beside the source.",
            "Final image remains legible in both original and translated areas.",
        ],
        "failure_traps": [
            "OCR on mirrored input before flipping.",
            "Over-sharpening before OCR and making characters noisy.",
            "Placing the translation over the evidence region.",
        ],
        "hard_negative_actions": [
            "Using OCR on the raw image.",
            "Applying the final ImageSR before Overlayer.",
            "Editing the whole image globally instead of isolating a translation panel.",
        ],
        "reward_focus": ["orientation dependency", "text recovery", "translation block layout", "late-stage preservation"],
        "why_hard": "This is a long dependency chain where geometry, readability, translation relayout, and late enhancement all interact.",
        "variables": {
            "language": ["Chinese", "Japanese", "French", "Spanish", "Arabic"],
        },
    },
    {
        "task_type": "cross_image_object_label_update_story_chain",
        "difficulty": "ultra",
        "cross_image": True,
        "required_tools": ["ImageGrounding", "ImageSAM", "ImageExtract", "OCR", "ImageCrop", "ImageEdit", "Overlayer", "ImageSR", "Overlayer"],
        "preferred_tool_order": ["ImageGrounding", "ImageSAM", "ImageExtract", "OCR", "ImageCrop", "ImageEdit", "Overlayer", "ImageSR", "Overlayer"],
        "scene_templates": [
            "product bottle photo, blank ad layout, and lifestyle background scene",
            "coffee cup source image, empty flyer template, and cafe counter destination",
            "sports bib photo, clean promo card, and stadium backdrop",
            "street sign source shot, poster mockup, and destination travel scene",
        ],
        "prompt_templates": [
            "Across the {scene}, extract the text-bearing hero object from the source, recover its label text, rebuild that label to '{replacement_text}', place the updated object into the clean layout, sharpen the layout result, then overlay that finished layout into the destination scene.",
            "Use the {scene} as a three-stage transfer: isolate the main object, read and update its visible label to '{replacement_text}', composite the updated object into the ad layout, polish the layout, and finally place the completed layout into the destination environment.",
            "From the {scene}, lift the primary label-bearing object, recover and replace its text with '{replacement_text}', assemble it into the blank layout, sharpen only after that layout is stable, and then insert the full layout into the destination scene.",
        ],
        "constraints": [
            "Object extraction must happen before label rewriting.",
            "Label update should remain attached to the object surface.",
            "Layout composition should stabilize before the destination-scene transfer.",
            "Destination placement should preserve believable scale and layer order.",
        ],
        "success_criteria": [
            "Correct object is isolated from the source.",
            "Visible label is updated locally on the object surface.",
            "Intermediate layout looks coherent before final transfer.",
            "Destination scene integration remains plausible after final overlay.",
        ],
        "failure_traps": [
            "Treating the label as floating text disconnected from the object.",
            "Polishing the layout too early before object and label are stable.",
            "Placing the updated object directly into the destination without building the intermediate layout.",
        ],
        "hard_negative_actions": [
            "Using Overlayer on the destination scene before the ad layout exists.",
            "Skipping OCR and guessing the original label text structure.",
            "Applying ImageSR before the intermediate layout is complete.",
        ],
        "reward_focus": ["three-stage transfer", "surface text attachment", "intermediate layout quality", "final scene realism"],
        "why_hard": "This forces the agent to plan across source extraction, intermediate layout building, and final destination transfer without collapsing steps together.",
        "variables": {
            "replacement_text": ["CITY PASS", "LIMITED DROP", "SEASON MENU", "NIGHT RACE", "OPEN WEEK"],
        },
    },
    {
        "task_type": "fine_grained_multi_instance_storyboard_chain",
        "difficulty": "expert",
        "cross_image": False,
        "required_tools": ["ImageGrounding", "ImageCrop", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "ImageGrounding", "Overlayer"],
        "preferred_tool_order": ["ImageGrounding", "ImageCrop", "ImageSAM", "ImageExtract", "ImageEdit", "Overlayer", "ImageGrounding", "Overlayer"],
        "scene_templates": [
            "crowded market display with repeated fruit labels",
            "busy desk with several similar notebooks and mugs",
            "kitchen shelf with repeated jars and bottles",
            "cosmetics shelf with near-identical containers",
        ],
        "prompt_templates": [
            "In the {scene}, select the {ordinal} {target_object}, isolate and edit it to have {attribute_value} {attribute}, place that edited object into a clean side panel, then highlight the original source instance in the main scene with a subtle marker so the before-versus-source relation is unambiguous.",
            "From the {scene}, identify the exact {ordinal} {target_object}, extract it, locally edit its {attribute} to {attribute_value}, display the edited result in a side panel, and mark the original instance in the cluttered source image with a subtle locator overlay.",
            "Use the {scene} to run a two-view task: isolate the {ordinal} {target_object}, create an edited version with {attribute_value} {attribute}, show that edited version separately, and add a subtle source marker on the original instance inside the scene.",
        ],
        "constraints": [
            "Ordinal instance selection must be exact.",
            "Edited side-panel object should remain recognizably linked to the source instance.",
            "Source marker must highlight the original object without obscuring it.",
            "The side panel and source scene should coexist clearly in one final layout.",
        ],
        "success_criteria": [
            "Correct source instance is targeted.",
            "Edited variant changes only the intended local attribute.",
            "Side panel cleanly displays the edited object.",
            "Original instance is clearly but subtly indicated in the source scene.",
        ],
        "failure_traps": [
            "Selecting the wrong near-duplicate instance.",
            "Marking the wrong object in the source after editing the right one.",
            "Using an overly strong source marker that hides visual evidence.",
        ],
        "hard_negative_actions": [
            "Editing before exact instance isolation.",
            "Skipping the second grounding step for source-marker placement.",
            "Building the final layout before the edited object exists.",
        ],
        "reward_focus": ["exact instance tracking", "edit locality", "source-result linkage", "final layout clarity"],
        "why_hard": "The model must keep track of the same instance across extraction, editing, side-panel display, and source-scene marking.",
        "variables": {
            "ordinal": ["leftmost", "second-from-left", "second-from-right", "rightmost", "front-most"],
            "target_object": ["red mug", "blue notebook", "green jar", "yellow bottle", "striped container"],
            "attribute": ["label accent", "rim color", "stripe pattern", "cap color", "surface finish"],
            "attribute_value": ["deep navy", "copper", "matte cream", "warm orange", "black stripe"],
        },
    },
    {
        "task_type": "weather_text_restore_dual_surface_chain",
        "difficulty": "extreme",
        "cross_image": False,
        "required_tools": ["OCR", "ImageGrounding", "ImageCrop", "ImageEdit", "ImageSAM", "ImageExtract", "Overlayer", "ImageSR"],
        "preferred_tool_order": ["OCR", "ImageGrounding", "ImageCrop", "ImageEdit", "ImageSAM", "ImageExtract", "Overlayer", "ImageSR"],
        "scene_templates": [
            "bookstore sidewalk scene with sign and poster",
            "street cafe terrace with menu board and window sticker",
            "bus stop ad scene with printed poster and bench label",
            "market stall with hanging sign and package labels",
        ],
        "prompt_templates": [
            "Turn the {scene} into {weather_target}, but preserve two different text-bearing surfaces: first recover what those surfaces say, then apply the weather edit, repair or reinsert any damaged text areas locally, and sharpen only after both surfaces are stable.",
            "For the {scene}, change the atmosphere to {weather_target} while keeping both the main sign and the secondary printed surface readable; OCR them first, perform the global weather edit, then locally restore whichever text-bearing surfaces need repair before a final clarity pass.",
            "Make the {scene} feel like {weather_target}, but protect two separate readable surfaces. Recover their wording first, edit the scene globally, then locally repair or reapply damaged text areas and only then enhance final clarity.",
        ],
        "constraints": [
            "Both text-bearing surfaces must be considered, not just the most prominent one.",
            "Global weather change should not be reverted when doing local text repair.",
            "Text restoration must remain confined to damaged surface regions.",
            "Final sharpening should not create halos around repaired text edges.",
        ],
        "success_criteria": [
            "Weather shift is globally visible.",
            "Primary and secondary text surfaces remain readable or are restored locally.",
            "Local repairs stay attached to their original surfaces.",
            "Final image remains coherent across atmosphere and readability goals.",
        ],
        "failure_traps": [
            "Saving only the main sign while forgetting the secondary printed surface.",
            "Undoing the weather effect around repaired text regions.",
            "Sharpening restored surfaces until they look pasted on.",
        ],
        "hard_negative_actions": [
            "Applying the weather edit before recovering what the text says.",
            "Repairing text globally instead of surface-locally.",
            "Running ImageSR before local text restoration is finished.",
        ],
        "reward_focus": ["global-local coordination", "multi-surface preservation", "attachment realism", "late enhancement discipline"],
        "why_hard": "This couples one global edit with two separate local text-preservation obligations, forcing the agent to track multiple fragile regions through the chain.",
        "variables": {
            "weather_target": ["light rain at dusk", "foggy morning", "fresh snowfall", "overcast drizzle", "golden hour after rain"],
        },
    },
    {
        "task_type": "layout_rebuild_comparison_translation_chain",
        "difficulty": "expert",
        "cross_image": False,
        "required_tools": ["OCR", "ImageGrounding", "ImageCrop", "ImageEdit", "Overlayer", "OCR", "Overlayer", "ImageSR"],
        "preferred_tool_order": ["OCR", "ImageGrounding", "ImageCrop", "ImageEdit", "Overlayer", "OCR", "Overlayer", "ImageSR"],
        "scene_templates": [
            "concert poster with dense hierarchy",
            "museum flyer with title, date, and sponsor footer",
            "festival one-sheet with multiple text blocks",
            "summit poster with schedule snippet and headline",
        ],
        "prompt_templates": [
            "Analyze the {scene}, replace only the main title and date with content about {topic}, keep the rest of the hierarchy stable, then OCR the final poster again and add a compact translated summary strip in {language} along the bottom before a late sharpen pass.",
            "Use the {scene} as a structured poster-edit task: localize the title and date blocks, swap them for {topic}, preserve the remaining layout, re-read the final poster content, and place a short {language} summary strip at the bottom only after the main layout is stable.",
            "For the {scene}, edit just the title and date to match {topic}, preserve layout and secondary text, then OCR the edited poster and add a small {language} summary strip across the bottom before the final clarity step.",
        ],
        "constraints": [
            "Title/date replacement must not disturb secondary layout blocks.",
            "The summary strip should be derived from the final edited poster, not the original one.",
            "Bottom summary strip must remain compact and not cover important footer content.",
            "Final sharpening should not disturb already-corrected text alignment.",
        ],
        "success_criteria": [
            "Only target title/date blocks are edited.",
            "Poster hierarchy remains recognizable after local changes.",
            "Final OCR reflects the edited version rather than the original.",
            "Summary strip is grounded in the edited poster content and placed cleanly.",
        ],
        "failure_traps": [
            "Generating the summary strip from stale pre-edit OCR.",
            "Changing the full poster instead of local blocks.",
            "Covering footer information with the bottom strip.",
        ],
        "hard_negative_actions": [
            "Skipping the second OCR pass after edits.",
            "Building the summary strip before the main poster edit is complete.",
            "Using ImageSR before final text additions are done.",
        ],
        "reward_focus": ["targeted layout editing", "post-edit verification", "summary grounding", "hierarchy preservation"],
        "why_hard": "The model must edit part of the layout, then verify the edited result again and use the verified content to drive a final summary overlay.",
        "variables": {
            "topic": ["a jazz weekend", "a documentary premiere", "a design workshop", "a food market", "an indie film retrospective"],
            "language": ["Chinese", "Japanese", "French", "Spanish", "Arabic"],
        },
    },
]


def pick(rng: random.Random, items: list[str]) -> str:
    """Pick one item."""
    return rng.choice(items)


def instantiate_prompt(rng: random.Random, blueprint: dict) -> tuple[str, list[str]]:
    """Create one unique prompt and its concrete constraints."""
    variables = {"scene": pick(rng, blueprint["scene_templates"])}
    for key, values in blueprint.get("variables", {}).items():
        variables[key] = pick(rng, values)

    base_prompt = pick(rng, blueprint["prompt_templates"]).format(**variables)
    extra_count = 3 if blueprint["difficulty"] == "expert" else 4
    extras = rng.sample(GLOBAL_QUALIFIERS, k=extra_count)
    delivery = pick(rng, DELIVERY_NOTES)
    user_prompt = f"{base_prompt} {' '.join(extras)} {delivery}"
    constraints = rng.sample(blueprint["constraints"], k=min(4, len(blueprint["constraints"])))
    return user_prompt, constraints


def build_sample(sample_id: int, blueprint: dict, rng: random.Random) -> dict:
    """Build one v3 sample."""
    user_prompt, constraints = instantiate_prompt(rng, blueprint)
    return {
        "id": f"rl_batch3_{sample_id:04d}",
        "difficulty": blueprint["difficulty"],
        "task_type": blueprint["task_type"],
        "cross_image": blueprint["cross_image"],
        "user_prompt": user_prompt,
        "required_tools": blueprint["required_tools"],
        "preferred_tool_order": blueprint["preferred_tool_order"],
        "min_tool_count": len(blueprint["required_tools"]),
        "long_chain_steps": len(blueprint["preferred_tool_order"]),
        "constraints": constraints,
        "why_hard": blueprint["why_hard"],
        "success_criteria": blueprint["success_criteria"],
        "failure_traps": rng.sample(blueprint["failure_traps"], k=min(2, len(blueprint["failure_traps"]))),
        "hard_negative_actions": rng.sample(blueprint["hard_negative_actions"], k=min(2, len(blueprint["hard_negative_actions"]))),
        "reward_focus": blueprint["reward_focus"],
        "tool_order_risk": "Very high: multiple downstream steps depend on early geometry/localization/extraction decisions.",
        "termination_condition": "Stop only when all sub-goals are satisfied, later polishing has not broken earlier local fixes, and untouched context still looks stable.",
    }


def blueprints_by_difficulty() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for blueprint in TASK_BLUEPRINTS:
        grouped.setdefault(blueprint["difficulty"], []).append(blueprint)
    return grouped


def generate_samples() -> list[dict]:
    rng = random.Random(RNG_SEED)
    grouped = blueprints_by_difficulty()
    seen_prompts: set[str] = set()
    samples: list[dict] = []
    sample_id = 1

    for difficulty, target in DIFFICULTY_TARGETS.items():
        blueprints = grouped[difficulty]
        count = 0
        attempts = 0
        while count < target:
            blueprint = blueprints[count % len(blueprints)]
            sample = build_sample(sample_id, blueprint, rng)
            attempts += 1
            if sample["user_prompt"] in seen_prompts:
                if attempts < target * 80:
                    continue
                sample["user_prompt"] += " Final validation should confirm every intermediate dependency was respected."
            seen_prompts.add(sample["user_prompt"])
            samples.append(sample)
            sample_id += 1
            count += 1
    return samples


def write_jsonl(samples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def render_markdown(samples: list[dict]) -> str:
    lines = [
        "# RL Batch 1000 V3 Ultra Preview",
        "",
        "Third batch with longer chains, more cross-image work, and higher order sensitivity.",
        "",
        "## Tool List",
        "",
    ]
    for tool_name, tool_kind in TOOL_INFO.items():
        lines.append(f"- `{tool_name}`: {tool_kind}")
    lines.extend(["", "## Samples", ""])

    for sample in samples:
        lines.append(f"### {sample['id']} | {sample['difficulty']} | {sample['task_type']}")
        lines.append(f"- Cross image: {sample['cross_image']}")
        lines.append(f"- User prompt: {sample['user_prompt']}")
        lines.append(f"- Required tools: {', '.join(sample['required_tools'])}")
        lines.append(f"- Preferred tool order: {' -> '.join(sample['preferred_tool_order'])}")
        lines.append(f"- Min tool count: {sample['min_tool_count']}")
        lines.append(f"- Long chain steps: {sample['long_chain_steps']}")
        lines.append(f"- Why hard: {sample['why_hard']}")
        lines.append(f"- Constraints: {'; '.join(sample['constraints'])}")
        lines.append(f"- Success criteria: {'; '.join(sample['success_criteria'])}")
        lines.append(f"- Failure traps: {'; '.join(sample['failure_traps'])}")
        lines.append(f"- Hard negative actions: {'; '.join(sample['hard_negative_actions'])}")
        lines.append(f"- Reward focus: {', '.join(sample['reward_focus'])}")
        lines.append(f"- Termination: {sample['termination_condition']}")
        lines.append("")
    return "\n".join(lines)


def summarize(samples: list[dict]) -> dict[str, object]:
    difficulty = Counter(sample["difficulty"] for sample in samples)
    task_types = Counter(sample["task_type"] for sample in samples)
    avg_tools = sum(sample["min_tool_count"] for sample in samples) / len(samples)
    avg_steps = sum(sample["long_chain_steps"] for sample in samples) / len(samples)
    cross_image = sum(1 for sample in samples if sample["cross_image"])
    long_chain_8_plus = sum(1 for sample in samples if sample["long_chain_steps"] >= 8)
    return {
        "difficulty": dict(difficulty),
        "task_types": len(task_types),
        "avg_tools": avg_tools,
        "avg_steps": avg_steps,
        "cross_image": cross_image,
        "long_chain_8_plus": long_chain_8_plus,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    samples = generate_samples()
    write_jsonl(samples, base / OUTPUT_JSONL)
    (base / OUTPUT_MD).write_text(render_markdown(samples), encoding="utf-8")
    summary = summarize(samples)
    print(f"generated={len(samples)}")
    print(f"difficulty={json.dumps(summary['difficulty'], ensure_ascii=False, sort_keys=True)}")
    print(f"task_types={summary['task_types']}")
    print(f"avg_tools={summary['avg_tools']:.2f}")
    print(f"avg_steps={summary['avg_steps']:.2f}")
    print(f"cross_image={summary['cross_image']}")
    print(f'long_chain_8_plus={summary["long_chain_8_plus"]}')


if __name__ == "__main__":
    main()
