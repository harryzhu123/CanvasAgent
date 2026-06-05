#!/usr/bin/env python3
"""Generate a 2000-sample long-chain RL batch with high R+C+T demands.

This batch targets the "long sequence calling" bucket:
- R: tasks require visual reasoning, recovery, or evidence-grounded planning
- C: tasks decompose into multiple dependent sub-goals
- T: tasks exercise diverse tools across perception, geometry, editing, and composition
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path


OUTPUT_JSONL = "rl_batch_2000_v5_long_chain.jsonl"
OUTPUT_MD = "rl_batch_2000_v5_long_chain_preview.md"
OUTPUT_STAGE12_JSON = "rl_batch_2000_v5_long_chain_stage12_keypoints.json"
RNG_SEED = 20260408 + 57
SYSTEM_PROMPT_PATH = Path("/nfsdata4/zhuhairui/zhuhairui/data/smartagentV2/for-cluster/systemprompt.txt")
DATA_SOURCE = "rl-batch5"
AGENT_NAME = "tool_agent"
ABILITY = "vision"

REWARD_TOOL_NAME_MAP = {
    "ImageGeneration": "Generation",
    "ImageEdit": "Edit",
}

DIFFICULTY_TARGETS = {
    "expert": 400,
    "extreme": 700,
    "ultra": 900,
}

GLOBAL_QUALIFIERS = [
    "Treat intermediate assets as evaluation targets rather than disposable scratch results.",
    "Do not collapse several local obligations into one vague global edit.",
    "Preserve untouched context, spatial logic, and text attachment while solving the main objective.",
    "Assume a reviewer will inspect whether every downstream step was enabled by correct earlier work.",
    "Avoid shortcuts that bypass evidence recovery, region isolation, or explicit layout construction.",
    "Keep object identity, perspective, and surface attachment stable through the full chain.",
    "If a step depends on readable evidence or exact localization, resolve that dependency first.",
    "Use the minimum global change necessary and handle fragile regions with local operations.",
]

DELIVERY_NOTES = [
    "Terminate only after every sub-goal is visually satisfied and later polish has not broken earlier fixes.",
    "Do not end the task while any extracted asset, title block, or destination placement still looks provisional.",
    "Final verification should check both the edited targets and the supposedly untouched supporting context.",
    "Only stop when the intermediate artifact would make sense on its own and the final scene uses it correctly.",
]

TOOL_INFO = {
    "ImageGeneration": "generation",
    "ImageEdit": "editing",
    "Crop": "cropping",
    "Rotate": "rotation",
    "Flip": "flipping",
    "Grounding": "grounding",
    "SAM": "masking",
    "Extract": "extraction",
    "SR": "super-resolution",
    "Overlayer": "overlay",
    "OCR": "ocr",
}

PROMPT_GENERATION_NOTES = {
    "style": "natural_end_user_request",
    "mention_tools": False,
    "must_preserve_dependency_order": True,
    "must_keep_constraints_explicit": True,
    "must_sound_actionable": True,
}

TASK_BLUEPRINTS = [
    {
        "task_type": "triple_image_sign_rebuild_transfer_chain",
        "difficulty": "ultra",
        "cross_image": True,
        "reasoning_source": [
            "must recover text evidence before translation and reconstruction",
            "must localize the true sign surface instead of nearby clutter",
            "must build a valid intermediate sign before destination transfer",
        ],
        "decomposition_targets": [
            "recover the original sign content",
            "reconstruct the translated sign on a clean carrier",
            "transfer the finished sign into a destination scene",
        ],
        "required_tools": ["Grounding", "Crop", "SR", "OCR", "SAM", "Extract", "ImageEdit", "Overlayer"],
        "preferred_tool_order": ["Grounding", "Crop", "SR", "OCR", "SAM", "Extract", "ImageEdit", "Overlayer"],
        "scene_templates": [
            "source storefront photo, blank sign panel, and rainy street destination",
            "mall sign photo, neutral acrylic plate, and subway corridor ad frame",
            "bookshop facade image, clean replacement lightbox, and nighttime avenue scene",
            "cafe sign source, empty menu board, and airport corridor billboard scene",
            "bakery window sign photo, blank hanging board, and pedestrian lane destination",
            "museum entrance plaque photo, clean brass panel, and indoor lobby wayfinding wall",
            "boutique facade sign image, matte sign cassette, and wet alley destination",
            "hotel reception sign shot, empty wall plaque, and upscale hallway destination",
            "theater marquee crop, neutral poster lightbox, and evening square scene",
            "farmers market booth sign photo, blank wooden insert, and outdoor plaza destination",
        ],
        "prompt_templates": [
            "Across the {scene}, recover the readable sign content from the source, translate it into {language}, isolate the true sign-bearing surface, rebuild the translated sign on the clean panel, then place that finished panel into the destination while preserving reflections, perspective, and local realism.",
            "Use the {scene} as a three-stage transfer task: identify and recover the physical sign in the source view, translate the message into {language}, reconstruct it on the blank panel, and finally insert the completed panel into the destination scene without breaking highlight logic or layer order.",
            "Work through the {scene}: localize the actual signage, enhance only enough to read it, translate the recovered wording into {language}, rebuild the sign on the replacement panel, and transfer that rebuilt asset into the destination scene with believable geometry and attachment.",
            "Treat the {scene} as a long-chain signage migration problem: first recover the source sign content, then translate it into {language}, rebuild the translated design on the clean carrier, and only after that move the finished asset into the destination scene with correct perspective and reflections.",
            "For the {scene}, isolate the real sign surface in the source image, recover and translate its message into {language}, construct the translated sign on the empty panel, and place that completed panel into the destination so it reads like part of the original environment rather than a pasted patch.",
            "Solve the {scene} in stages: find and recover the physical signage, translate the recovered text into {language}, rebuild the result on the blank panel, and integrate that panel into the destination while keeping highlight direction, scale, and layer order believable.",
            "In the {scene}, do not edit the destination first. Recover the source sign content, translate it into {language}, create a finished translated sign on the clean panel, and then transfer that intermediate asset into the destination with realistic attachment and untouched surrounding context.",
            "Use the {scene} to recover a damaged or distant sign, translate the recovered wording into {language}, reconstruct the sign on the replacement panel, and embed the finished panel into the destination only after the intermediate sign already looks complete on its own.",
            "Across the {scene}, read the source sign accurately enough to support a {language} version, rebuild that translated sign on the clean board, and insert the completed board into the destination scene without breaking perspective, local lighting, or reflective cues.",
            "Handle the {scene} as a source-to-panel-to-destination pipeline: recover the original sign, translate it into {language}, rebuild it on the blank panel, and then deploy that panel into the destination scene while keeping the final result physically consistent.",
        ],
        "constraints": [
            "Text recovery must happen before translation or sign reconstruction.",
            "The extracted source asset must correspond to the true sign surface, not adjacent facade details.",
            "The reconstructed sign should be completed on the clean panel before the destination transfer.",
            "Destination insertion must preserve perspective, reflection, and scale cues.",
        ],
        "success_criteria": [
            "Readable source sign content is recovered correctly enough to guide translation.",
            "The translated sign is cleanly reconstructed on the blank panel.",
            "The final scene contains the rebuilt sign with coherent perspective and layer order.",
            "Untouched destination context remains plausible around the inserted sign.",
        ],
        "failure_traps": [
            "Translating guessed sign content instead of recovered text evidence.",
            "Editing the final destination globally before the intermediate sign exists.",
            "Extracting a surrounding wall patch instead of the actual sign surface.",
        ],
        "hard_negative_actions": [
            "Calling ImageEdit on the destination scene before reconstructing the sign panel.",
            "Skipping OCR and inventing the source wording.",
            "Trying to place the source crop directly into the destination without rebuilding the intermediate sign.",
        ],
        "reward_focus": ["evidence recovery", "intermediate asset construction", "cross-image transfer", "perspective realism"],
        "why_hard": "The chain is long because recovery, translation, reconstruction, and destination transfer each create a dependency for the next stage.",
        "variables": {
            "language": ["Chinese", "Japanese", "French", "Spanish", "Arabic"],
        },
    },
    {
        "task_type": "multi_image_identity_cover_chain",
        "difficulty": "ultra",
        "cross_image": True,
        "reasoning_source": [
            "must choose a visually stable source identity from multiple photos",
            "must separate subject extraction from title-card composition",
            "must ground any final title in readable evidence from the source set",
        ],
        "decomposition_targets": [
            "identify the best source instance",
            "build a clean hero cutout",
            "compose the subject onto a blank cover and add evidence-grounded titling",
        ],
        "required_tools": ["Grounding", "Crop", "SR", "SAM", "Extract", "Overlayer", "OCR", "Overlayer"],
        "preferred_tool_order": ["Grounding", "Crop", "SR", "SAM", "Extract", "Overlayer", "OCR", "Overlayer"],
        "scene_templates": [
            "an event portrait set of the same athlete plus a blank magazine cover",
            "a travel portrait set of the same guide plus a clean book jacket",
            "a backstage musician portrait set plus a neutral poster cover",
            "a conference speaker portrait set plus an empty keynote opener slide",
            "a restaurant chef portrait set plus a clean menu cover",
            "a street interview portrait set plus a neutral magazine page",
            "a gallery docent portrait set plus a blank exhibition booklet cover",
            "a coach sideline portrait set plus a clean sports program cover",
            "a teacher classroom portrait set plus an empty school brochure cover",
            "a host studio portrait set plus a minimal livestream thumbnail card",
        ],
        "prompt_templates": [
            "From the {scene}, choose the strongest source view for subject identity, enhance only the details needed to stabilize that choice, extract one clean hero subject, place it onto the empty cover, then recover any visible name text from the source set and add it as a restrained title after the main composition is stable.",
            "Use the {scene} to build a cover image: identify the source photo that best preserves the person's geometry, refine the evidence enough to support a clean extraction, create a hero cutout, place it onto the blank cover, and then overlay the subject's visible name only if it can be read from the source images.",
            "Across the {scene}, resolve identity first, isolate the best hero subject, transfer that subject onto the clean cover, then OCR the most readable name evidence from the source set and place a tidy title only after the cover composition is finalized.",
            "Treat the {scene} as an identity-selection task: decide which source frame preserves the person most cleanly, stabilize the details needed for a precise extraction, place that hero subject on the blank cover, and only then add any name text that is readable in the source set.",
            "For the {scene}, avoid blending multiple identities together. Pick the strongest source portrait, refine it enough to support a clean cutout, compose that subject onto the cover, and add a subtle title only if the name can be read from the original images.",
            "Work through the {scene} by first committing to one identity-consistent source image, extracting a clean hero figure from it, placing that figure on the empty cover, and then recovering any visible name evidence from the photo set for a restrained final title.",
            "In the {scene}, select the source image with the best face geometry and edge quality, use it to build a clean hero cutout, transfer that cutout to the blank cover, and add name text only if it is visibly readable in the source images.",
            "Solve the {scene} as a source-set consolidation problem: choose the best portrait for identity, extract one polished hero subject, place it onto the cover card, and then pull any readable name text from the original photos into a minimal cover title.",
            "Use the {scene} to create a coherent cover without identity drift: identify the best source frame, enhance just enough to support extraction, place the final hero cutout on the clean cover, and overlay a title only when the source images provide readable name evidence.",
            "Across the {scene}, lock in the most reliable source identity, create one clean extracted subject from that evidence, transfer the subject to the empty cover, and finish with source-evidence-based titling only if the source set visibly contains a name.",
        ],
        "constraints": [
            "The cover subject should come from one coherent source view rather than a stitched identity.",
            "Text overlay should be added only after the hero composition is visually stable.",
            "Any displayed name must be grounded in readable source evidence, not guessed.",
            "Hairline and shoulder boundaries should remain clean in the extracted hero subject.",
        ],
        "success_criteria": [
            "One stable hero subject is selected and extracted from the source set.",
            "The hero subject is composited cleanly onto the cover.",
            "Any title text is justified by OCR-readable evidence from the photos.",
            "The final cover looks intentional rather than assembled from conflicting source cues.",
        ],
        "failure_traps": [
            "Mixing incompatible face angles across different photos into one unstable hero.",
            "Adding a guessed name with no readable evidence.",
            "Placing title text before the cover composition is settled.",
        ],
        "hard_negative_actions": [
            "Running OCR before deciding which photo has the best readable name evidence.",
            "Using Overlayer to place the subject before extraction quality is acceptable.",
            "Solving the task with a global edit across all source photos instead of selecting one coherent identity source.",
        ],
        "reward_focus": ["identity consistency", "clean extraction", "cover composition", "evidence-grounded titling"],
        "why_hard": "The model must reason over multiple source images, commit to one identity-consistent extraction, and only then build the final cover with evidence-grounded text.",
    },
    {
        "task_type": "curved_label_recover_replace_reinsert_chain",
        "difficulty": "ultra",
        "cross_image": False,
        "reasoning_source": [
            "must decide the minimum geometric correction needed for readability",
            "must isolate the true curved label surface before local rewriting",
            "must preserve surface attachment when reinserting the edited patch",
        ],
        "decomposition_targets": [
            "recover readable text from a curved or warped label",
            "extract the label-bearing surface",
            "rewrite and reinsert the patch without flattening the object",
        ],
        "required_tools": ["Rotate", "Grounding", "SR", "OCR", "SAM", "Extract", "ImageEdit", "Overlayer"],
        "preferred_tool_order": ["Rotate", "Grounding", "SR", "OCR", "SAM", "Extract", "ImageEdit", "Overlayer"],
        "scene_templates": [
            "drink cup sleeve with bent lettering",
            "curved cosmetic bottle label",
            "helmet sticker photo with arced text",
            "arched banner wrapped around a sports stand",
            "rounded candle jar label seen in perspective",
            "curved snack canister with warped branding",
            "bike water bottle with wraparound text",
            "paper coffee bag photographed from an angle",
            "plastic shampoo bottle with side-curved label",
            "ceramic bowl sleeve with bent wording",
        ],
        "prompt_templates": [
            "In the {scene}, apply only the minimum rotation needed to make the wording readable, localize the true label area, recover the text, isolate the curved label surface, replace the wording with '{replacement_text}', and reinsert that edited patch so the final surface still looks physically curved and attached.",
            "Use the {scene} for a recover-and-reinsert task: make the label readable without over-correcting the geometry, identify the exact text-bearing surface, read it, extract the local patch, rewrite it to '{replacement_text}', and place it back without turning the object into a flat sticker.",
            "For the {scene}, recover warped text through minimal geometric correction, isolate the actual label patch, replace the wording with '{replacement_text}', and return the edited patch to the object so the curvature and material continuity remain believable.",
            "Treat the {scene} as a curved-surface rewrite problem: correct the view only enough to read the label, isolate the actual text-bearing surface, replace the wording with '{replacement_text}', and reattach the edited patch so the object still looks naturally curved.",
            "With the {scene}, first make the original wording legible, then extract the true label patch, rewrite it to '{replacement_text}', and place it back in a way that preserves the object's three-dimensional wrapping and edge continuity.",
            "Solve the {scene} in order: minimal rotation for readability, precise localization of the label, OCR-guided replacement to '{replacement_text}', and reinsertion of the edited patch so the final result still follows the object's curvature.",
            "In the {scene}, do not flatten the object just to read it. Recover the label content, isolate the curved surface patch, replace the wording with '{replacement_text}', and return that patch to the object with believable material attachment.",
            "Use the {scene} to recover distorted label text, pinpoint the exact curved patch that carries it, rewrite the patch to '{replacement_text}', and reinsert the edited result so the object still reads as a wrapped surface rather than a flat decal.",
            "Handle the {scene} as a read-extract-rewrite-reinsert chain: recover the original label, isolate the correct curved region, change it to '{replacement_text}', and restore it to the object without introducing flat geometry or pasted seams.",
            "For the {scene}, make the wording readable through the smallest geometric correction possible, extract the label-bearing surface, replace it with '{replacement_text}', and blend the edited patch back so the final object keeps its original curvature and texture logic.",
        ],
        "constraints": [
            "Readability recovery should not become the final flattened geometry.",
            "Localization must target the real label surface rather than decorative neighbors.",
            "Replacement text should inherit the original curved attachment logic.",
            "Reinserted edges should blend back into the original object without obvious seams.",
        ],
        "success_criteria": [
            "The original wording becomes readable enough to guide the replacement.",
            "The correct curved label patch is isolated and edited.",
            "The new wording sits naturally on the curved surface.",
            "The final object still looks three-dimensional rather than patched with a flat sticker.",
        ],
        "failure_traps": [
            "Editing text directly on the unreadable raw surface without recovery.",
            "Extracting a broad flat area and losing the label's true attachment.",
            "Using the rotated recovery image itself as the final output.",
        ],
        "hard_negative_actions": [
            "Skipping OCR and guessing the original wording structure.",
            "Trying to write new text globally across the whole object.",
            "Stopping after the recovery view instead of reinserting the edited patch.",
        ],
        "reward_focus": ["minimal correction", "surface localization", "patch reintegration", "geometry preservation"],
        "why_hard": "Each stage creates a hard dependency: if the recovery view, label localization, or patch extraction is wrong, the reinsertion stage cannot look correct.",
        "variables": {
            "replacement_text": ["CITY PASS", "LIMITED DROP", "NIGHT RIDE", "SUMMER SET", "ROUTE 24"],
        },
    },
    {
        "task_type": "document_reverse_recover_translate_summary_chain",
        "difficulty": "extreme",
        "cross_image": False,
        "reasoning_source": [
            "must resolve mirror and orientation issues before OCR",
            "must identify the useful text region instead of editing the whole document",
            "must translate the main message while preserving source evidence",
        ],
        "decomposition_targets": [
            "correct reading direction",
            "recover the main message",
            "add a translated summary block without covering evidence",
        ],
        "required_tools": ["Flip", "Rotate", "Crop", "SR", "OCR", "ImageEdit", "Overlayer", "SR"],
        "preferred_tool_order": ["Flip", "Rotate", "Crop", "SR", "OCR", "ImageEdit", "Overlayer", "SR"],
        "scene_templates": [
            "mirrored event notice photographed at an angle",
            "reversed menu board seen through glass",
            "tilted transit notice reflected in a shop window",
            "crooked bilingual instruction card captured in a selfie shot",
            "backlit building directory reflected in a glossy panel",
            "side-view cafe schedule board shot through a window",
            "reversed museum info placard photographed obliquely",
            "mirrored pop-up store poster on a shiny wall",
            "skewed hotel welcome notice reflected in a door panel",
            "crooked safety instruction sheet seen in a mirror",
        ],
        "prompt_templates": [
            "Take the {scene}, fix the mirrored and tilted reading direction first, isolate the important text region, enhance it enough to recover the main message, translate that message into {language}, place the translation as a clean side summary block, and sharpen only after the full layout is stable.",
            "For the {scene}, undo reversal and angle errors before OCR, crop to the useful text-bearing area, improve legibility, recover the main message, translate it into {language}, and add a compact translated summary beside the source without covering the evidence region.",
            "Recover the {scene} into readable form, extract the main text content, translate it into {language}, build a summary panel next to the original document, and reserve sharpening for the final step only.",
            "Treat the {scene} as an orientation-dependent recovery task: correct the mirrored or skewed reading direction, isolate the main text region, recover the message, translate it into {language}, and place that translation in a clean adjacent block only after the source is readable.",
            "Use the {scene} to restore a difficult document view: fix reversal and tilt first, crop to the evidence-bearing area, enhance the text enough for OCR, translate the main message into {language}, and add a compact summary panel without covering the original proof.",
            "Solve the {scene} in sequence by correcting the reading direction, recovering the main text region, translating the recovered content into {language}, and laying out a side summary that preserves the original source evidence.",
            "In the {scene}, do not translate before the source becomes readable. First resolve mirror and angle issues, recover the central message, then add a {language} summary block beside the source and keep final sharpening for the very end.",
            "Handle the {scene} as a recover-translate-relayout pipeline: correct the orientation, isolate the useful text, OCR the main message, translate it into {language}, and position the translated block so the original document remains visible for comparison.",
            "For the {scene}, bring the source into readable orientation, extract the most important text content, convert that main message into {language}, and place the translated summary next to the source without disturbing the original evidence area.",
            "Across the {scene}, first solve geometry, then solve language: correct reversal and tilt, recover the message, translate it into {language}, and add a side summary only after the source text has become legible enough to trust.",
        ],
        "constraints": [
            "Reading-direction fixes must happen before OCR.",
            "The summary block must not occlude the original evidence region.",
            "Only the main message should be translated, not unrelated decorative text.",
            "Late sharpening must preserve already-corrected text alignment and readability.",
        ],
        "success_criteria": [
            "Mirroring and tilt are corrected before text recovery.",
            "The main message becomes readable enough to transcribe and translate.",
            "The translated summary block is placed cleanly beside the source document.",
            "Both source and translated areas remain legible in the final result.",
        ],
        "failure_traps": [
            "Running OCR on mirrored input before correcting direction.",
            "Placing the translated block on top of the evidence region.",
            "Sharpening too early and amplifying noise before OCR succeeds.",
        ],
        "hard_negative_actions": [
            "Using OCR on the raw reversed image.",
            "Adding the translated panel before the recovered content is known.",
            "Applying the final SR before the summary block exists.",
        ],
        "reward_focus": ["orientation dependency", "text recovery", "translation layout", "late-stage preservation"],
        "why_hard": "Geometry, readability, translation, and layout have to be solved in the right order, or the entire chain breaks down.",
        "variables": {
            "language": ["Chinese", "Japanese", "French", "Spanish", "Arabic"],
        },
    },
    {
        "task_type": "weather_dual_surface_restore_chain",
        "difficulty": "extreme",
        "cross_image": False,
        "reasoning_source": [
            "must identify more than one fragile text-bearing surface before a global edit",
            "must preserve recovered wording through a strong atmosphere change",
            "must decide which regions require local repair after the global pass",
        ],
        "decomposition_targets": [
            "recover the readable surfaces",
            "apply a global weather or lighting transformation",
            "repair damaged local regions while preserving the new atmosphere",
        ],
        "required_tools": ["OCR", "Grounding", "Crop", "ImageEdit", "SAM", "Extract", "Overlayer", "SR"],
        "preferred_tool_order": ["OCR", "Grounding", "Crop", "ImageEdit", "SAM", "Extract", "Overlayer", "SR"],
        "scene_templates": [
            "bookstore sidewalk with a sign and a poster",
            "street cafe terrace with a menu board and window sticker",
            "bus stop ad scene with a printed poster and bench label",
            "market stall with a hanging sign and product card",
            "cinema entrance with a marquee board and ticket notice",
            "museum lobby with a wall title and brochure stand card",
            "train platform with a schedule board and carriage-side label",
            "food truck exterior with a menu panel and side sticker",
            "campus bulletin area with a notice board and directional placard",
            "hotel exterior with a facade sign and door-hours decal",
        ],
        "prompt_templates": [
            "Turn the {scene} into {weather_target}, but preserve two separate text-bearing surfaces: recover their wording first, perform the global atmosphere change, then locally repair or reapply whichever surfaces lost readability while keeping the new weather effect intact, and finish with a final clarity pass.",
            "For the {scene}, read the important sign and secondary printed surface before changing the whole scene to {weather_target}; after the global edit, restore any damaged text-bearing regions locally and only then enhance final clarity.",
            "Make the {scene} feel like {weather_target}, but do not sacrifice the readable sign and secondary printed surface. Recover both first, apply the global change, repair the fragile regions locally, and reserve final polishing for the end.",
            "Treat the {scene} as a global-local coordination problem: identify the two readable surfaces first, shift the entire atmosphere to {weather_target}, then repair only the surfaces that lost legibility while keeping the new weather visible everywhere else.",
            "Use the {scene} to perform a strong environment edit to {weather_target}, but first recover both the main sign and the secondary printed surface, then locally restore any damaged regions after the global pass and finish with a restrained clarity enhancement.",
            "Solve the {scene} in two layers: preserve the readable surfaces by recovering them first, transform the full scene into {weather_target}, and only afterward repair whichever local regions need help without undoing the broader atmospheric change.",
            "In the {scene}, do not protect only the biggest sign. Recover the main and secondary text surfaces first, apply the {weather_target} look to the whole scene, then restore local readability where needed while preserving the new lighting and weather mood.",
            "Handle the {scene} as a weather transformation with two fragile obligations: read both text-bearing surfaces, convert the scene to {weather_target}, repair local damage selectively, and leave the final image coherent across atmosphere and readability.",
            "For the {scene}, change the full environment to {weather_target} only after understanding what both text surfaces say, then use local restoration to keep those surfaces readable without erasing the newly created weather effects.",
            "Across the {scene}, first secure the sign and the secondary printed region as evidence, then perform the {weather_target} transformation, and finally repair only the regions that need local help before a last clarity pass.",
        ],
        "constraints": [
            "Both text-bearing surfaces must be tracked, not just the main sign.",
            "The global atmosphere change should remain visible after local restoration.",
            "Text repairs must stay confined to the damaged regions.",
            "Final sharpening should not create pasted-on halos around repaired text edges.",
        ],
        "success_criteria": [
            "The weather or lighting shift is visible across the full scene.",
            "Primary and secondary text surfaces remain readable or are restored locally.",
            "Repaired regions still look attached to their original surfaces.",
            "The final image stays coherent across atmosphere and readability goals.",
        ],
        "failure_traps": [
            "Recovering only the main sign and forgetting the second fragile surface.",
            "Undoing the new weather around repaired text regions.",
            "Using aggressive local repair that makes surfaces look pasted back on.",
        ],
        "hard_negative_actions": [
            "Applying the global ImageEdit before reading the original text evidence.",
            "Repairing text with another global scene edit instead of local operations.",
            "Running SR before the local restoration stage is complete.",
        ],
        "reward_focus": ["global-local coordination", "multi-surface preservation", "attachment realism", "post-edit repair"],
        "why_hard": "The model has to protect multiple fragile regions through one strong global change and then restore only what needs local intervention.",
        "variables": {
            "weather_target": ["light rain at dusk", "foggy morning", "fresh snowfall", "overcast drizzle", "golden hour after rain"],
        },
    },
    {
        "task_type": "poster_rewrite_reverify_summary_chain",
        "difficulty": "extreme",
        "cross_image": False,
        "reasoning_source": [
            "must identify exactly which text blocks should change and which should stay",
            "must use a second verification pass after the main rewrite",
            "must derive the summary strip from the edited poster rather than stale content",
        ],
        "decomposition_targets": [
            "recover the original layout hierarchy",
            "edit only the target blocks",
            "verify the edited result and add a compact summary strip",
        ],
        "required_tools": ["OCR", "Grounding", "Crop", "ImageEdit", "Overlayer", "OCR", "Overlayer", "SR"],
        "preferred_tool_order": ["OCR", "Grounding", "Crop", "ImageEdit", "Overlayer", "OCR", "Overlayer", "SR"],
        "scene_templates": [
            "concert poster with dense text hierarchy",
            "museum flyer with title, date, and sponsor footer",
            "festival one-sheet with multiple text blocks",
            "summit poster with headline, date line, and small agenda block",
            "charity gala poster with title, date, and partner footer",
            "book launch flyer with author line, date, and venue block",
            "workshop notice with headline, date line, and registration footer",
            "film screening poster with title, date, and credit block",
            "fashion pop-up poster with headline, location, and sponsor line",
            "community fair flyer with main title, date, and organizer footer",
        ],
        "prompt_templates": [
            "Analyze the {scene}, replace only the main title and date with content about {topic}, preserve the rest of the hierarchy, then OCR the edited poster again and add a compact {language} summary strip along the bottom before the final clarity pass.",
            "Use the {scene} as a structured poster-edit task: localize the title and date blocks, rewrite them for {topic}, keep secondary information stable, re-read the edited poster, and overlay a short {language} summary strip only after the main layout is correct.",
            "For the {scene}, change just the title and date to match {topic}, preserve the remaining hierarchy, verify the edited poster with a second OCR pass, and add a clean {language} bottom summary strip before the last enhancement step.",
            "Treat the {scene} as a selective layout rewrite: identify the title and date blocks, update only those blocks for {topic}, confirm the edited version with a second OCR pass, and then add a compact {language} summary strip that reflects the final poster rather than the original.",
            "Use the {scene} to edit a poster without disturbing its hierarchy: localize the primary title and date, rewrite them around {topic}, verify the finished poster by reading it again, and add a short {language} summary strip only after the main composition is settled.",
            "Solve the {scene} in stages by preserving the full layout structure, replacing only the title and date with {topic}, re-reading the edited result, and placing a restrained {language} summary strip across the bottom before final enhancement.",
            "In the {scene}, do not rewrite the whole poster. Change only the key title and date information to fit {topic}, run OCR again on the edited version, and use that verified result to create a clean {language} summary strip at the bottom.",
            "Handle the {scene} as a targeted poster revision: update the headline and date for {topic}, keep the remaining blocks intact, verify the new state with a second OCR pass, and place a compact {language} summary strip without covering important footer information.",
            "For the {scene}, first understand the original layout hierarchy, then replace only the main title and date for {topic}, read the edited poster again, and add a concise {language} summary strip grounded in that final edited content.",
            "Across the {scene}, preserve the poster structure while changing just the title and date to {topic}, verify the rewritten poster through OCR, and then add a bottom {language} summary strip that reflects the new version rather than stale source text.",
        ],
        "constraints": [
            "Only the title and date blocks should change; secondary layout should remain stable.",
            "The summary strip must be derived from the edited poster, not the original version.",
            "The summary strip should remain compact and avoid covering important footer content.",
            "Final sharpening must not disturb corrected text alignment or spacing.",
        ],
        "success_criteria": [
            "Target title and date blocks are rewritten while the rest of the layout stays recognizable.",
            "A second OCR pass reflects the edited poster content.",
            "The summary strip is grounded in the edited version and placed cleanly.",
            "The final poster retains hierarchy, readability, and alignment.",
        ],
        "failure_traps": [
            "Changing the entire poster instead of the target blocks.",
            "Building the summary strip from stale OCR taken before the edits.",
            "Covering sponsor or footer information with an oversized bottom strip.",
        ],
        "hard_negative_actions": [
            "Skipping the second OCR pass after editing.",
            "Adding the summary strip before the main poster rewrite is stable.",
            "Using SR before all text additions are finished.",
        ],
        "reward_focus": ["targeted layout editing", "post-edit verification", "summary grounding", "hierarchy preservation"],
        "why_hard": "This chain requires a rewrite, a verification pass on the new state, and a final addition that depends on the verified state rather than the original one.",
        "variables": {
            "topic": ["a jazz weekend", "a documentary premiere", "a design workshop", "a food market", "an indie film retrospective"],
            "language": ["Chinese", "Japanese", "French", "Spanish", "Arabic"],
        },
    },
    {
        "task_type": "generated_product_comparison_chain",
        "difficulty": "expert",
        "cross_image": False,
        "reasoning_source": [
            "must turn a generated scene into reusable assets rather than editing it monolithically",
            "must preserve object identity between the original and modified variant",
            "must ground the caption in visible generated text instead of inventing it",
        ],
        "decomposition_targets": [
            "generate a source scene",
            "extract and locally modify the hero object",
            "assemble a comparison layout with an evidence-grounded caption",
        ],
        "required_tools": ["ImageGeneration", "Grounding", "SAM", "Extract", "ImageEdit", "Overlayer", "OCR", "Overlayer"],
        "preferred_tool_order": ["ImageGeneration", "Grounding", "SAM", "Extract", "ImageEdit", "Overlayer", "OCR", "Overlayer"],
        "scene_templates": [
            "product poster with one hero bottle, a large title banner, and accessory props",
            "sports promo card with one athlete, a headline strip, and repeated equipment",
            "food advertisement with a main bowl, a menu title panel, and side dishes",
            "travel card layout with a hero sign, a visible title block, and decorative icons",
            "tech gadget launch poster with one device and a bold heading area",
            "skincare ad with one hero tube and a simple brand banner",
            "beverage promo board with one can and a headline ribbon",
            "outdoor gear poster with one backpack and a title panel",
            "stationery ad with one notebook and a visible top slogan area",
            "toy product card with one hero figure and a label header",
        ],
        "prompt_templates": [
            "Generate a {scene}, isolate the main hero object, create a modified version with {attribute_value} {attribute}, place the original and edited versions into a side-by-side comparison panel, then OCR any readable generated title or label text and add it as a restrained comparison caption.",
            "Create a {scene}, separate the hero object from the generated background, make a local variant with {attribute_value} {attribute}, build a before-versus-after panel from those two object states, and recover any readable generated title text as a short caption.",
            "Make a {scene}; after generation, extract the hero object, edit only its {attribute} to {attribute_value}, assemble the original and edited versions into a clean comparison strip, then read any visible title or label text and add it as a compact caption line.",
            "Treat the {scene} as a generate-then-recompose task: create the base image, extract the hero object, make a {attribute_value} {attribute} variant, present original and edited states side by side, and use any readable generated title element as the final caption.",
            "Use the {scene} to produce a comparison card by generating the source layout first, isolating the hero object, editing its {attribute} to {attribute_value}, arranging both versions in one panel, and OCR-ing any readable title or sign text from the generated scene for the caption.",
            "Solve the {scene} in steps: generate the composition, separate the hero object, build a modified version with {attribute_value} {attribute}, assemble a before-versus-after layout, and then recover any readable generated heading as the caption rather than inventing one.",
            "In the {scene}, avoid editing the full generated background. Extract the hero object, create a local {attribute_value} {attribute} variant, place both object states into a comparison layout, and use any readable generated title text as a compact caption.",
            "Handle the {scene} as a reusable-asset problem: generate the full poster, isolate the hero object, create a targeted {attribute_value} {attribute} change, build a comparison strip, and OCR whatever readable headline or label the generated image provides for the caption.",
            "For the {scene}, first create the source image, then separate and edit the hero object to have {attribute_value} {attribute}, show the original and modified states together, and recover any readable generated title element to anchor the caption.",
            "Across the {scene}, generate the source layout, preserve the hero object's identity while changing its {attribute} to {attribute_value}, build a side-by-side comparison, and caption it using only text that is actually readable in the generated image.",
        ],
        "constraints": [
            "The edited variant must remain recognizably the same object as the original.",
            "The comparison layout should clearly separate original and edited states.",
            "Caption text must come from visible generated evidence rather than invention.",
            "The background scene should not be globally damaged while creating the comparison panel.",
        ],
        "success_criteria": [
            "A coherent source scene is generated.",
            "The hero object is correctly isolated and edited locally.",
            "The final comparison layout clearly presents original versus modified states.",
            "The caption is supported by OCR-visible generated text.",
        ],
        "failure_traps": [
            "Editing the whole generated scene instead of the hero object.",
            "Losing object identity between the original and edited versions.",
            "Guessing a caption instead of reading available text from the scene.",
        ],
        "hard_negative_actions": [
            "Using ImageEdit before a clean hero object exists.",
            "Building the comparison panel before both object states are available.",
            "Running OCR before the generated scene has any readable headline evidence.",
        ],
        "reward_focus": ["generation reliability", "object identity", "comparison clarity", "caption grounding"],
        "why_hard": "The task demands generation, asset extraction, controlled local editing, layout assembly, and evidence-grounded captioning in a single chain.",
        "variables": {
            "attribute": ["trim", "label accent", "surface finish", "stripe pattern", "badge color"],
            "attribute_value": ["copper", "forest green", "matte black", "warm orange", "deep blue"],
        },
    },
    {
        "task_type": "fine_grained_instance_trace_chain",
        "difficulty": "expert",
        "cross_image": False,
        "reasoning_source": [
            "must select the exact intended instance among near-duplicates",
            "must keep track of the same instance through extraction and side-panel display",
            "must separately mark the original source instance after editing the extracted copy",
        ],
        "decomposition_targets": [
            "identify the correct source instance",
            "extract and modify the target object",
            "show the edited result while linking it back to the source scene",
        ],
        "required_tools": ["Grounding", "Crop", "SAM", "Extract", "ImageEdit", "Overlayer", "Grounding", "Overlayer"],
        "preferred_tool_order": ["Grounding", "Crop", "SAM", "Extract", "ImageEdit", "Overlayer", "Grounding", "Overlayer"],
        "scene_templates": [
            "cluttered tabletop with repeated mugs, notebooks, jars, bottles, and containers",
            "busy studio shelf with several similar desk objects and pantry items",
            "shared office counter with repeated notebooks, bottles, cups, and storage jars",
            "kitchen workbench with mixed jars, bottles, mugs, and containers",
            "craft table with several similar cups, notebooks, jars, and bottles",
            "retail display shelf with mixed small containers, bottles, mugs, and journals",
            "classroom supply table with repeated notebooks, cups, bottles, and plastic jars",
            "backstage prep counter with mixed drink bottles, mugs, notebooks, and cases",
            "market checkout shelf with assorted jars, bottles, cups, and boxed containers",
            "home entry console with repeated drinkware, notebooks, jars, and organizers",
        ],
        "prompt_templates": [
            "In the {scene}, find the {ordinal} {target_object}, isolate it, change its {attribute} to {attribute_value}, place the edited result into a clean side panel, then mark the original source instance subtly so the source-to-result link is unambiguous.",
            "From the {scene}, identify the exact {ordinal} {target_object}, extract it, locally edit its {attribute} to {attribute_value}, present the edited version in a side panel, and add a subtle source marker to the original object in the crowded scene.",
            "Use the {scene} to run a source-trace task: select the {ordinal} {target_object}, create an edited variant with {attribute_value} {attribute}, show that variant separately, and lightly indicate the original source instance without obscuring it.",
            "Treat the {scene} as an instance-tracking problem: locate the {ordinal} {target_object}, isolate that exact instance, modify its {attribute} to {attribute_value}, display the edited result in a side panel, and then mark the original source item with a subtle locator.",
            "For the {scene}, resolve the correct {ordinal} {target_object} before doing any edits, extract it cleanly, change its {attribute} to {attribute_value}, show the edited version separately, and indicate the original source object without hiding the evidence.",
            "Solve the {scene} in order by identifying the {ordinal} {target_object}, extracting and editing it to have {attribute_value} {attribute}, placing the edited object into a side display, and then marking the exact source instance inside the original scene.",
            "In the {scene}, do not confuse near-duplicates. Track down the {ordinal} {target_object}, create an edited version with {attribute_value} {attribute}, show that result in a clean side panel, and add a restrained source marker to the same object in the original view.",
            "Handle the {scene} as a select-edit-trace chain: choose the {ordinal} {target_object}, isolate it, update its {attribute} to {attribute_value}, place the edited object in a side panel, and lightly mark the source instance so the linkage stays obvious.",
            "Use the {scene} to identify the exact {ordinal} {target_object}, preserve its identity through extraction, edit its {attribute} to {attribute_value}, present the edited result separately, and then highlight the matching source object with a subtle visual marker.",
            "Across the {scene}, first resolve which object is the {ordinal} {target_object}, then extract and edit only that item to have {attribute_value} {attribute}, show the transformed result in a side panel, and mark the original instance without obscuring its appearance.",
        ],
        "constraints": [
            "Ordinal or spatial selection must target the exact intended instance.",
            "The edited side-panel object should remain recognizably linked to the source instance.",
            "The source marker must be subtle enough to preserve the original visual evidence.",
            "The final layout should clearly contain both the source scene and the edited side result.",
        ],
        "success_criteria": [
            "The correct near-duplicate instance is selected.",
            "The extracted object is edited only in the requested local attribute.",
            "The side panel cleanly displays the edited object.",
            "The source marker points to the same instance that produced the side-panel result.",
        ],
        "failure_traps": [
            "Editing the right object but marking the wrong source instance afterward.",
            "Selecting the wrong duplicate because the ordinal cue is ignored.",
            "Using an overly strong source marker that hides the evidence.",
        ],
        "hard_negative_actions": [
            "Editing before exact instance isolation is complete.",
            "Skipping the second grounding step for source-marker placement.",
            "Building the final panel before the edited object exists.",
        ],
        "reward_focus": ["instance disambiguation", "edit locality", "source-result linkage", "layout clarity"],
        "why_hard": "The challenge is not just local editing; it is persistent instance tracking across extraction, transformation, and source-scene reference.",
        "variables": {
            "ordinal": ["leftmost", "second-from-left", "middle", "second-from-right", "rightmost"],
            "target_object": ["red mug", "blue notebook", "green jar", "yellow bottle", "striped container"],
            "attribute": ["label accent", "rim color", "stripe pattern", "cap color", "surface finish"],
            "attribute_value": ["deep navy", "copper", "matte cream", "warm orange", "black stripe"],
        },
    },
]


def validate_blueprints() -> None:
    """Sanity-check blueprint template counts and basic uniqueness."""
    for blueprint in TASK_BLUEPRINTS:
        scene_templates = blueprint["scene_templates"]
        prompt_templates = blueprint["prompt_templates"]
        task_type = blueprint["task_type"]

        if len(scene_templates) != 10:
            raise ValueError(f"{task_type}: expected 10 scene_templates, got {len(scene_templates)}")
        if len(prompt_templates) != 10:
            raise ValueError(f"{task_type}: expected 10 prompt_templates, got {len(prompt_templates)}")
        if len(set(scene_templates)) != len(scene_templates):
            raise ValueError(f"{task_type}: duplicate entries found in scene_templates")
        if len(set(prompt_templates)) != len(prompt_templates):
            raise ValueError(f"{task_type}: duplicate entries found in prompt_templates")


def pick(rng: random.Random, items: list[str]) -> str:
    """Pick one item from a list."""
    return rng.choice(items)


def instantiate_prompt(rng: random.Random, blueprint: dict) -> tuple[str, list[str], dict]:
    """Create one unique prompt and its concrete constraints."""
    variables = {"scene": pick(rng, blueprint["scene_templates"])}
    for key, values in blueprint.get("variables", {}).items():
        variables[key] = pick(rng, values)

    base_prompt = pick(rng, blueprint["prompt_templates"]).format(**variables)
    extra_count = 3 if blueprint["difficulty"] == "expert" else 4
    qualifiers = rng.sample(GLOBAL_QUALIFIERS, k=extra_count)
    delivery = pick(rng, DELIVERY_NOTES)
    user_prompt = f"{base_prompt} {' '.join(qualifiers)} {delivery}"
    constraint_count = min(4, len(blueprint["constraints"]))
    constraints = rng.sample(blueprint["constraints"], k=constraint_count)
    return user_prompt, constraints, variables


def build_sample(sample_id: int, blueprint: dict, rng: random.Random) -> dict:
    """Build one sample in the v5 long-chain batch."""
    user_prompt, constraints, variables = instantiate_prompt(rng, blueprint)
    return {
        "id": f"rl_batch5_{sample_id:04d}",
        "difficulty": blueprint["difficulty"],
        "task_type": blueprint["task_type"],
        "cross_image": blueprint["cross_image"],
        "instantiated_variables": variables,
        "reasoning_source": blueprint["reasoning_source"],
        "decomposition_targets": blueprint["decomposition_targets"],
        "user_prompt": user_prompt,
        "required_tools": blueprint["required_tools"],
        "preferred_tool_order": blueprint["preferred_tool_order"],
        "min_tool_count": len(blueprint["required_tools"]),
        "long_chain_steps": len(blueprint["preferred_tool_order"]),
        "tool_diversity_count": len(set(blueprint["required_tools"])),
        "constraints": constraints,
        "why_hard": blueprint["why_hard"],
        "success_criteria": blueprint["success_criteria"],
        "failure_traps": rng.sample(blueprint["failure_traps"], k=min(2, len(blueprint["failure_traps"]))),
        "hard_negative_actions": rng.sample(blueprint["hard_negative_actions"], k=min(2, len(blueprint["hard_negative_actions"]))),
        "reward_focus": blueprint["reward_focus"],
        "tool_order_risk": "Extremely high: reading, localization, extraction, and composition errors propagate through the rest of the chain.",
        "termination_condition": "Stop only when every sub-goal is complete, intermediate assets have been consumed correctly, and final polishing has not damaged earlier edits.",
    }


def blueprints_by_difficulty() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for blueprint in TASK_BLUEPRINTS:
        grouped.setdefault(blueprint["difficulty"], []).append(blueprint)
    return grouped


def generate_samples() -> list[dict]:
    validate_blueprints()
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
                if attempts < target * 120:
                    continue
                sample["user_prompt"] += " Final verification should explicitly confirm that each intermediate dependency was respected."
            seen_prompts.add(sample["user_prompt"])
            samples.append(sample)
            sample_id += 1
            count += 1

    return samples


def load_system_prompt() -> str:
    """Load the system prompt used for RL10k-like exports."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def normalize_tool_names(tool_names: list[str]) -> list[str]:
    """Map tool names into the RL10k-style reward schema."""
    return [REWARD_TOOL_NAME_MAP.get(name, name) for name in tool_names]


def infer_input_mode(sample: dict) -> str:
    """Infer whether the final user request should be prompt-only or require input images."""
    first_tool = sample["required_tools"][0] if sample["required_tools"] else None

    if first_tool == "ImageGeneration" and not sample["cross_image"]:
        return "prompt_only"
    if sample["task_type"] in {"triple_image_sign_rebuild_transfer_chain", "multi_image_identity_cover_chain"}:
        return "multi_image_prompt_pair"
    return "single_image_prompt_pair"


def split_scene_components(scene: str) -> list[str]:
    """Split a cross-image scene description into coarse asset components."""
    normalized = re.sub(r",\s+and\s+", ", ", scene)
    normalized = re.sub(r"\s+plus\s+", ", ", normalized)
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return parts


def build_task_goal(sample: dict) -> str:
    """Summarize the intended task in one planning-oriented sentence."""
    task_type = sample["task_type"]
    variables = sample["instantiated_variables"]
    scene = variables["scene"]

    if task_type == "triple_image_sign_rebuild_transfer_chain":
        language = variables["language"]
        return (
            f"Use the {scene} to recover a real sign, translate its content into {language}, "
            "rebuild it on a clean carrier, and transfer that rebuilt asset into a destination scene."
        )
    if task_type == "multi_image_identity_cover_chain":
        return (
            f"Use {scene} to choose one identity-consistent source portrait, extract a clean hero subject, "
            "and build a cover that may include a name copied from readable source evidence."
        )
    if task_type == "curved_label_recover_replace_reinsert_chain":
        replacement_text = variables["replacement_text"]
        return (
            f"Use the {scene} to recover warped label text, replace it with '{replacement_text}', "
            "and reinsert the edited patch while preserving the curved attachment."
        )
    if task_type == "document_reverse_recover_translate_summary_chain":
        language = variables["language"]
        return (
            f"Use the {scene} to correct orientation, recover the main readable message, translate it into {language}, "
            "and add a compact translated summary block without hiding the evidence."
        )
    if task_type == "weather_dual_surface_restore_chain":
        weather_target = variables["weather_target"]
        return (
            f"Use the {scene} to preserve two fragile text-bearing surfaces through a global transformation to {weather_target}, "
            "then locally restore any regions that lose readability."
        )
    if task_type == "poster_rewrite_reverify_summary_chain":
        topic = variables["topic"]
        language = variables["language"]
        return (
            f"Use the {scene} to rewrite only the title and date around {topic}, verify the edited result again, "
            f"and add a compact {language} summary strip based on the final poster."
        )
    if task_type == "generated_product_comparison_chain":
        attribute = variables["attribute"]
        attribute_value = variables["attribute_value"]
        return (
            f"Use the {scene} to generate a source layout, extract the hero object, change its {attribute} to {attribute_value}, "
            "and assemble a comparison card with caption text grounded in visible evidence."
        )
    if task_type == "fine_grained_instance_trace_chain":
        ordinal = variables["ordinal"]
        target_object = variables["target_object"]
        attribute = variables["attribute"]
        attribute_value = variables["attribute_value"]
        return (
            f"Use the {scene} to find the {ordinal} {target_object}, edit only its {attribute} to {attribute_value}, "
            "show the edited result separately, and mark the original source instance."
        )

    return f"Solve a long-chain multimodal task grounded in the scene: {scene}."


def build_asset_role_hints(sample: dict) -> list[dict]:
    """Describe the input images that a later LLM planner may need to request or synthesize."""
    task_type = sample["task_type"]
    variables = sample["instantiated_variables"]
    scene = variables["scene"]

    if task_type == "triple_image_sign_rebuild_transfer_chain":
        parts = split_scene_components(scene)
        source = parts[0] if len(parts) > 0 else "source storefront photo"
        panel = parts[1] if len(parts) > 1 else "blank sign panel"
        destination = parts[2] if len(parts) > 2 else "destination street scene"
        return [
            {
                "slot": "img_1",
                "role": "source_sign_image",
                "asset_description": source,
                "must_contain": ["one physical sign surface", "recoverable real-world text"],
                "must_avoid": ["completely blank sign", "heavy blur that destroys OCR"],
                "why_needed": "Provides the source sign content and surface evidence.",
            },
            {
                "slot": "img_2",
                "role": "blank_rebuild_carrier",
                "asset_description": panel,
                "must_contain": ["clean editable sign surface", "minimal geometry ambiguity"],
                "must_avoid": ["existing logos", "existing readable text"],
                "why_needed": "Acts as the intermediate carrier for the rebuilt translated sign.",
            },
            {
                "slot": "img_3",
                "role": "destination_scene",
                "asset_description": destination,
                "must_contain": ["plausible placement area", "coherent lighting and perspective"],
                "must_avoid": ["no valid placement surface", "visually incompatible scale"],
                "why_needed": "Receives the finished rebuilt sign in the final transfer step.",
            },
        ]

    if task_type == "multi_image_identity_cover_chain":
        parts = split_scene_components(scene)
        source_desc = parts[0] if len(parts) > 0 else "portrait set of the same person"
        cover_desc = parts[-1] if len(parts) > 1 else "blank cover card"
        portrait_roles = [
            ("img_1", "primary_portrait", "clear front-facing identity reference"),
            ("img_2", "secondary_portrait", "three-quarter portrait of the same person"),
        ]
        assets = []
        for slot, role, desc in portrait_roles:
            assets.append(
                {
                    "slot": slot,
                    "role": role,
                    "asset_description": f"{source_desc} featuring {desc}",
                    "must_contain": ["the same identity", "clean enough extraction edges"],
                    "must_avoid": ["identity drift", "occlusions that hide the subject"],
                    "why_needed": "Provides candidate identity-consistent source images for hero selection.",
                }
            )
        assets.append(
            {
                "slot": "img_3",
                "role": "blank_cover",
                "asset_description": cover_desc,
                "must_contain": ["clean layout", "negative space for subject and title"],
                "must_avoid": ["existing person", "existing dominant title text"],
                "why_needed": "Acts as the final composition surface for the extracted hero subject.",
            }
        )
        return assets

    if task_type == "curved_label_recover_replace_reinsert_chain":
        return [
            {
                "slot": "img_1",
                "role": "curved_label_scene",
                "asset_description": scene,
                "must_contain": ["curved attached label", "warped but recoverable wording"],
                "must_avoid": ["perfectly flat label", "text-free object surface"],
                "why_needed": "Supports text recovery, curved-surface extraction, local rewrite, and reinsertion.",
            }
        ]

    if task_type == "document_reverse_recover_translate_summary_chain":
        return [
            {
                "slot": "img_1",
                "role": "mirrored_or_tilted_document",
                "asset_description": scene,
                "must_contain": ["one clear main text block", "mirror or tilt distortion"],
                "must_avoid": ["perfectly readable straight document", "no meaningful text"],
                "why_needed": "Provides a document that requires orientation recovery before OCR and translation.",
            }
        ]

    if task_type == "weather_dual_surface_restore_chain":
        return [
            {
                "slot": "img_1",
                "role": "weather_transform_scene",
                "asset_description": scene,
                "must_contain": ["two distinct readable text-bearing surfaces", "neutral clear-weather baseline"],
                "must_avoid": ["only one readable surface", "already extreme weather effects"],
                "why_needed": "Supports a global atmosphere change while preserving and restoring multiple fragile regions.",
            }
        ]

    if task_type == "poster_rewrite_reverify_summary_chain":
        return [
            {
                "slot": "img_1",
                "role": "editable_poster",
                "asset_description": scene,
                "must_contain": ["main title block", "date block", "smaller footer hierarchy"],
                "must_avoid": ["layout with no clear title-date separation", "illegible text hierarchy"],
                "why_needed": "Provides a poster whose main title and date can be selectively rewritten and re-verified.",
            }
        ]

    if task_type == "generated_product_comparison_chain":
        return []

    if task_type == "fine_grained_instance_trace_chain":
        target_object = variables["target_object"]
        target_noun = target_object.split()[-1]
        return [
            {
                "slot": "img_1",
                "role": "crowded_instance_scene",
                "asset_description": scene,
                "must_contain": [
                    f"multiple candidate {target_object}s so ordinal selection matters",
                    f"additional nearby distractor {target_noun}s and related objects",
                ],
                "must_avoid": ["only one obvious candidate", "empty uncluttered layout"],
                "why_needed": "Supports ordinal instance selection, extraction, side-panel display, and source-instance marking.",
            }
        ]

    return []


def build_key_variables(sample: dict) -> dict:
    """Keep only compact task-specific variables that matter for later prompt synthesis."""
    return {
        key: value
        for key, value in sample["instantiated_variables"].items()
        if key != "scene"
    }


def compact_asset_roles(sample: dict) -> list[dict]:
    """Shrink asset-role hints to the minimum fields needed for image-prompt planning."""
    compact_roles = []
    for role in build_asset_role_hints(sample):
        compact_roles.append(
            {
                "slot": role["slot"],
                "role": role["role"],
                "description": role["asset_description"],
                "must_contain": role["must_contain"],
                "must_avoid": role["must_avoid"],
            }
        )
    return compact_roles


def build_sign_transfer_assets(sample: dict) -> list[dict]:
    scene = sample["instantiated_variables"]["scene"]
    parts = split_scene_components(scene)
    source = parts[0] if len(parts) > 0 else "source storefront photo"
    panel = parts[1] if len(parts) > 1 else "blank sign panel"
    destination = parts[2] if len(parts) > 2 else "destination street scene"
    return [
        {
            "slot": "img_1",
            "image_prompt": f"Photorealistic {source} with one clear real-world sign containing readable text, realistic facade context, natural perspective, and no heavy motion blur.",
        },
        {
            "slot": "img_2",
            "image_prompt": f"Photorealistic {panel} with a clean blank editable surface, no visible text or logo, simple geometry, and realistic material texture.",
        },
        {
            "slot": "img_3",
            "image_prompt": f"Photorealistic {destination} with believable urban context, realistic lighting, and a plausible placement area for a rebuilt sign asset.",
        },
    ]


def build_cover_assets(sample: dict) -> list[dict]:
    scene = sample["instantiated_variables"]["scene"]
    parts = split_scene_components(scene)
    source_desc = parts[0] if len(parts) > 0 else "portrait set of the same person"
    cover_desc = parts[-1] if len(parts) > 1 else "blank cover card"
    variations = [
        "front-facing portrait with clear facial visibility and clean body outline",
        "three-quarter view portrait with a different pose but the same identity",
    ]
    assets = []
    for idx, variation in enumerate(variations, start=1):
        assets.append(
            {
                "slot": f"img_{idx}",
                "image_prompt": f"Photorealistic {source_desc} featuring the same person with consistent identity. Create one image showing a {variation}. Preserve realistic background context and clear extraction-friendly edges.",
            }
        )
    assets.append(
        {
            "slot": f"img_{len(assets) + 1}",
            "image_prompt": f"Minimal {cover_desc} with a clean blank layout, no person, no title text, and enough negative space for later subject placement and cover typography.",
        }
    )
    return assets


def build_single_image_asset(sample: dict) -> dict:
    task_type = sample["task_type"]
    variables = sample["instantiated_variables"]
    scene = variables["scene"]

    if task_type == "curved_label_recover_replace_reinsert_chain":
        image_prompt = (
            f"Photorealistic {scene} with curved or warped text physically attached to the object surface. "
            "The wording should be somewhat readable after correction, and the material, perspective, and lighting should look natural."
        )
    elif task_type == "document_reverse_recover_translate_summary_chain":
        image_prompt = (
            f"Photorealistic {scene}. The document should contain a clear main text block, but appear mirrored, reflected, tilted, or partially distorted so that orientation correction and OCR are necessary."
        )
    elif task_type == "weather_dual_surface_restore_chain":
        image_prompt = (
            f"Photorealistic {scene} in neutral clear weather. Include two distinct readable text-bearing surfaces, such as a main sign and a secondary poster, card, sticker, or printed label."
        )
    elif task_type == "poster_rewrite_reverify_summary_chain":
        image_prompt = (
            f"Front-facing graphic design poster matching {scene}, with a readable main title area, a distinct date line, and a smaller footer or sponsor block. Keep the layout structured and OCR-friendly."
        )
    elif task_type == "fine_grained_instance_trace_chain":
        target_object = variables["target_object"]
        target_noun = target_object.split()[-1]
        image_prompt = (
            f"Photorealistic {scene}. Include multiple candidate {target_object}s so ordinal selection matters, plus additional visually similar {target_noun}s and nearby distractors. Maintain realistic clutter, clear object boundaries, and natural lighting."
        )
    else:
        image_prompt = f"Photorealistic {scene} with realistic detail, clear structure, and natural lighting suitable for a multi-step visual tool-use task."

    return {"slot": "img_1", "image_prompt": image_prompt}


def build_input_assets(sample: dict) -> tuple[str, list[dict]]:
    """Infer input modality and create prompts for generating required source images."""
    input_mode = infer_input_mode(sample)

    if input_mode == "prompt_only":
        return "prompt_only", []
    if sample["task_type"] == "triple_image_sign_rebuild_transfer_chain":
        return "multi_image_prompt_pair", build_sign_transfer_assets(sample)
    if sample["task_type"] == "multi_image_identity_cover_chain":
        return "multi_image_prompt_pair", build_cover_assets(sample)
    return "single_image_prompt_pair", [build_single_image_asset(sample)]


def build_user_content(user_prompt: str, image_count: int) -> str:
    """Format the user turn with one <image> token per input image."""
    return f"{'<image>' * image_count}{user_prompt}" if image_count else user_prompt


def to_export_record(sample: dict, index: int, system_prompt: str) -> dict:
    """Project the full sample into an RL10k-like export schema."""
    input_mode, input_assets = build_input_assets(sample)
    reward_tools = normalize_tool_names(sample["required_tools"])
    return {
        "data_source": DATA_SOURCE,
        "agent_name": AGENT_NAME,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_content(sample["user_prompt"], len(input_assets))},
        ],
        "ability": ABILITY,
        "reward_model": {
            "ground_truth": reward_tools,
            "style": "rule",
        },
        "extra_info": {
            "index": index,
            "original_id": index,
            "tone": "explicit",
            "tool_path": reward_tools,
            "crossimage": sample["cross_image"],
            "inputmode": input_mode,
            "longchainsteps": sample["long_chain_steps"],
            "task_type": sample["task_type"],
            "whyhard": sample["why_hard"],
        },
        "images": input_assets,
    }


def to_stage12_record(sample: dict, index: int) -> dict:
    """Export one structured keypoint record for the later Qwen prompt-integration stage."""
    input_mode = infer_input_mode(sample)
    asset_roles = compact_asset_roles(sample)
    return {
        "index": index,
        "sample_id": sample["id"],
        "task_type": sample["task_type"],
        "cross_image": sample["cross_image"],
        "required_tools": sample["required_tools"],
        "toolchainlength": sample["long_chain_steps"],
        "scene": sample["instantiated_variables"]["scene"],
        "task_brief": build_task_goal(sample),
        "key_variables": build_key_variables(sample),
        "constraints": sample["constraints"],
        "style_requirements": PROMPT_GENERATION_NOTES,
        "input_spec": {
            "mode_hint": input_mode,
            "image_count": len(asset_roles),
            "asset_roles": asset_roles,
        },
        "output_spec": {
            "userprompt": "Generate one natural end-user instruction that preserves dependencies and constraints but does not mention tool names.",
            "image_prompt": "Return an empty list when no input image is needed; otherwise return one image prompt per required input image slot in order.",
        },
    }


def write_jsonl(samples: list[dict], path: Path) -> None:
    system_prompt = load_system_prompt()
    with path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(samples):
            f.write(json.dumps(to_export_record(sample, index, system_prompt), ensure_ascii=False) + "\n")


def write_stage12_json(samples: list[dict], path: Path) -> None:
    records = [to_stage12_record(sample, index) for index, sample in enumerate(samples)]
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def render_markdown(samples: list[dict]) -> str:
    lines = [
        "# RL Batch 2000 V5 Long-Chain Preview",
        "",
        "Fifth batch focused on high-R+C+T long-sequence tasks with explicit decomposition, high tool diversity, and strong order sensitivity.",
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
        lines.append(f"- Reasoning source: {'; '.join(sample['reasoning_source'])}")
        lines.append(f"- Decomposition targets: {'; '.join(sample['decomposition_targets'])}")
        lines.append(f"- Required tools: {', '.join(sample['required_tools'])}")
        lines.append(f"- Preferred tool order: {' -> '.join(sample['preferred_tool_order'])}")
        lines.append(f"- Min tool count: {sample['min_tool_count']}")
        lines.append(f"- Long chain steps: {sample['long_chain_steps']}")
        lines.append(f"- Tool diversity count: {sample['tool_diversity_count']}")
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
    avg_diversity = sum(sample["tool_diversity_count"] for sample in samples) / len(samples)
    cross_image = sum(1 for sample in samples if sample["cross_image"])
    eight_step = sum(1 for sample in samples if sample["long_chain_steps"] >= 8)
    high_diversity = sum(1 for sample in samples if sample["tool_diversity_count"] >= 7)
    return {
        "difficulty": dict(difficulty),
        "task_types": len(task_types),
        "avg_tools": avg_tools,
        "avg_steps": avg_steps,
        "avg_diversity": avg_diversity,
        "cross_image": cross_image,
        "eight_step": eight_step,
        "high_diversity": high_diversity,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    samples = generate_samples()
    write_stage12_json(samples, base / OUTPUT_STAGE12_JSON)
    write_jsonl(samples, base / OUTPUT_JSONL)
    (base / OUTPUT_MD).write_text(render_markdown(samples), encoding="utf-8")
    summary = summarize(samples)
    print(f"generated={len(samples)}")
    print(f"difficulty={json.dumps(summary['difficulty'], ensure_ascii=False, sort_keys=True)}")
    print(f"task_types={summary['task_types']}")
    print(f"avg_tools={summary['avg_tools']:.2f}")
    print(f"avg_steps={summary['avg_steps']:.2f}")
    print(f"avg_diversity={summary['avg_diversity']:.2f}")
    print(f"cross_image={summary['cross_image']}")
    print(f"eight_step={summary['eight_step']}")
    print(f"high_diversity={summary['high_diversity']}")


if __name__ == "__main__":
    main()
