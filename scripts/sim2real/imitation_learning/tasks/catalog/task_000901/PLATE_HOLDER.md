# Task000901 plate-holder scene

Variant: `Cyclo-Real-WhiteRoom-Task000901-PlateHolder-FFW-SG2-v0` (`--variant plate_holder`). The existing default cookware scene remains available. This variant loads the Task901 70cm table, SG2 reference pose and cameras, three colored plates and the measured wooden holder. It omits the default scene's cookware and produce, matching the new live 1050 reference photo.

The 2026-11-02 1050 camera capture is under `/home/robotis-ai/agents/experiments/2026-11-03_09-56-43_task000901_vision/object_capture/`. No hardware command was sent. The object centers are approximate tabletop projections of manually selected pixels using AprilTags 1 and 2 and the archived tag layout; see `layout_estimate.json`. This is a visual alignment, not a physical object-pose calibration.

## Measured holder

`runtime/data/objects/plate_holder_v1/wood_plate_holder.usda` has a 295 × 120 × 90 mm outer box. Two 295 × 17 × 17 mm long rails support seven 9 mm diameter pegs each. Peg centers are 45 mm apart; the resulting end-center margin is 12.5 mm, near the reported approximately 15 mm. The small cross braces are visible in the right wrist image, but their 10 mm diameter and exact position are visual estimates. The holder is a static visual/collision asset; mass and grasp behavior are not specified.

The holder is fixed in the SG2 variant at approximately `(-0.5580, 0.1658, 0.7002)` m with yaw 90 degrees. The earlier AprilTag and peg-tip fit was the starting point; this pose was adjusted against the current robot-mounted head and right-wrist renders. This is a visual placement, not a measured world pose. Its 18 original collision prims are retained, while 18 UV-mapped visual meshes use the photograph-guided `wood_grain_v2.png` material. `build_holder.py` reproduces the textured USD.

## Plates

`runtime/data/objects/plates_v1/{pastel_pink,light_blue,lime_yellow}.usda` are color-only references to the shared Task901 `yellow_plate.usda`. `build_thin_plate.py` generates that asset with Task000200's radial mesh/collision generator and a 2 mm vertical material thickness at every profile station. This interprets the user's "0.2" as 0.2 cm material thickness, pending confirmation of units and measurement location. The 190 mm diameter, 24 mm overall rim height and 160 convex collision sectors remain inherited geometry. The source USD retains its inherited 58 g mass, while the plate-holder scene explicitly overrides all three plates to the user-specified 50 g (0.050 kg) as of 2026-11-03. Earlier replay runs used the 58 g override. Diameter and overall height remain unmeasured estimates.

## Lighting and color

The SG2 `plate_holder` variant alone uses `white_room_plate_holder_lighting.usda` and `white_room_plate_holder.usda`. The four ceiling spots, dome, wall material and tabletop texture scale were adjusted against the archived 1050 head image. The three glossy plastic colors and the holder wood color were then tuned under those lights. This matches camera RGB appearance at selected image regions, not measured physical illuminance or calibrated surface reflectance. The default Task901 cookware scene and shared WhiteRoom assets are unaffected.

## Validation

The updated robot-camera comparison is at `/home/robotis-ai/agents/experiments/2026-11-03_09-56-43_task000901_vision/sim_camera_color/final/head_wrist_real_vs_sim.png`. Each partition panel is 50 cm wide with 3 mm seams. The revision 2 head camera uses a 0.96 focal-length multiplier with corrected roll/pitch to fit the table rear edge and gripper visibility. At three sampled image columns, the table-edge vertical error is now 1–2 px. The left wrist camera uses a smooth RGB tone response to retain shadow detail; the holder uses a more yellow texture tint `(0.85, 1.35, 1.05, 1)`. Both adjustments are in the scene/sensor, and the comparison compositor applies no color edits. `build_holder.py` preserves this tint when regenerating the asset. The captured real frames and joint state are reused without accessing or commanding the real robot. The final validation and image diagnostics are beside the comparison PNG. This is a static image fit; moving-arm and contact equivalence have not been validated.

## Revision 3 color refinement

The final comparison now uses three vertically stacked rows (head, left wrist, right wrist), each with real/sim columns. The left simulated camera gain is `(0.90, 0.92, 0.96)` with the existing gamma, smooth channel highlight recovery and saturation 1.08. Holder rails/end grain and peg side grain have separate measured appearance tints, `(0.82, 1.18, 0.86, 1)` and `(0.53, 0.84, 0.96, 1)`. The seven head peg samples changed from RGB `(143,130,82)` to `(111,99,79)` versus real `(107,98,80)`. The left wall is `(99,115,130)` versus real `(99,114,128)`. The `final/color_metrics.json` file records all chosen samples, including remaining front-rail and tabletop differences. Stored real references were reused; no physical robot access was made.

## Revision 4 subtle red tint

At user request, the holder rails/end grain now use `(0.88, 1.14, 0.86, 1)` and peg sides use `(0.57, 0.82, 0.96, 1)`. This is a small red increase over revision 3 to balance the head/right-wrist appearance. Camera settings and geometry are unchanged. The same saved real reference and three-row layout are used in `final/`, with previous evidence in `before_revision4/`.

## Revision 5 wrist red response

The user requested more R in both simulated wrists and less yellow in the right-wrist holder. The left R gain is now 0.97 (previously 0.90), and the right R gain is 0.94 (previously 0.86). The right sensor also adds `0.55 * max(G-B,0)` to B after gain to soften the yellow cast. Head camera, materials and geometry are unchanged. The right holder front-rail median changed from `(128,113,79)` to `(140,113,98)`, compared with saved real `(129,110,103)`. Parameters and comparison evidence are recorded in `revision5/` and `final/` of the camera-color experiment.

## 2026-11-03 initial plate alignment and dynamic contact check

The default plate XY positions now match the initial frame of episode 0 in `Task_000901_Put_Vegetable_In_Pot_And_Close_Lid_Leaderboard_MCAP`. Rim ellipse centers were fitted to the existing robot-mounted head projection at the recorded joint pose, retaining the 190 mm diameter and 24 mm rim height. The settled render differs by 0.19 px (blue), 0.61 px (pink), and 0.33 px (yellow) in head-image center. This is an image fit, not an independently measured 3D pose.

Plates are explicitly dynamic, gravity-enabled 58 g bodies with the existing 160 convex sectors each. Their contact material uses static/dynamic friction 0.6/0.45, zero restitution, and max friction combination with the existing gripper rubber. These are simulation parameters, not measured real coefficients. CCD and 240 Hz integration are enabled, with 32/4 plate position/velocity solver iterations. The task keeps a 60 Hz environment step via decimation 4.

A physics replay uses recorded joint positions as PD targets; original command streams are absent. Object poses are written only at initialization. The pink plate lifts about 18.5 cm, stays above 10 cm for about 1.8 seconds, then slips/drops. Holder placement does not succeed. Thus initial grasp/contact capability is demonstrated, but successful task replay is not. The updated head/wrist comparison and validation are under `/home/robotis-ai/agents/experiments/2026-11-03_09-56-43_task000901_real_sim_replay/episode_000/physics/`.

The 240 Hz configuration also passed a 5 cm drop check: peak transient plate/table penetration was 0.58 mm and all plates settled at z=0.700 m. The earlier 120 Hz trial had 2.62 mm penetration; its evidence is preserved under `validation_120hz/`.

## GPU CCD runtime capability clarification

During episode-5 replay on 2026-11-03, Isaac explicitly reported `CCD is not supported on GPU, ignoring request to enable it`. The configured scene/body CCD flags and the `ccd` field in older audits describe the request, not effective GPU CCD support. Actual replay uses 240 Hz GPU contact integration; the prior drop-test penetration measurements remain empirical results and must not be attributed to effective CCD. No physics parameters were changed for episode 5. See its `validation/runtime_capabilities.json` and renderer log.

## Episode 5 user specified 50 g plate mass

The shared plate-holder scene mass override changed from 0.058 kg to 0.050 kg for all three colors. The episode-5 rerun reuses the exact initial XY fit, recorded joint targets, cameras, contact coefficients and 240 Hz integration. The source USD mass stays 0.058 kg; the scene MassPropertiesCfg override is authoritative at spawn and the replay checks each effective PhysX mass. Results are in `episode_005/physics/mass_050g/` within the existing timestamped replay group. The preceding 58 g run remains at `episode_005/physics/`.

## Episode 5 flat base diameter 160 mm

At the user's request, the flat plate base diameter changed from 150 to 160 mm. Outer diameter remains 190 mm, overall height 24 mm and vertical material thickness 2 mm. The slope radii are compressed from the old 75–95 mm interval into 80–95 mm while retaining the height stations. Both visible bands and all 160 convex collision sectors are regenerated from this same profile. Color-only layers are preserved. Each plate remains 50 g, and the generator now also writes 0.050 kg into the source USD, superseding the inherited source-mass caveat above. The plate-holder scene mass override remains 0.050 kg.

The existing build_thin_plate.py under the grouped vision/object_capture folder now resolves the workspace root after folder reorganization and reproduces the profile and mass. Original geometry and generator snapshots are preserved in episode_005/physics/mass_050g_flat_160mm/input. Initial XY, cameras, trajectory, friction and 240 Hz integration are retained for the new replay.

## Episode 5 user specified 20 g plastic plates

The user identified the plates as plastic and requested 20 g each. The plate-holder scene override, Task901 source USD and build_thin_plate.py now specify 0.020 kg. Existing glossy plastic visual materials and rigid-body collision shapes are retained. Flat diameter stays 160 mm, outer diameter 190 mm, overall height 24 mm and vertical thickness 2 mm. Friction remains the previous simulation estimate; no measured plastic deformation model is claimed. The new replay reuses the same episode-5 initial XY and recorded joint targets and is stored in episode_005/physics/mass_020g_flat_160mm. Previous 50 g results remain preserved.

## Episode 6 with 15 g plates and 1.5 mm thickness

The user requested 15 g plastic plates for stored episode 6 and reduced material thickness from 2 to 1.5 mm (0.15 cm). Scene mass, source USD and generator now agree on 0.015 kg. The radial generator uses 1.5 mm vertical material at each station in visual and collision meshes. Flat diameter 160 mm, outer diameter 190 mm and outer upper rim height 24 mm are retained; the flat upper surface is now z=1.5 mm. Existing plastic appearance and estimated rigid contact material are retained. Initial plate XY is refitted to episode 6 visible rim arcs because the initial gripper partly occludes pink/yellow.

The lower-right tag 5 existed but projected entirely outside the head frame. Tag5Paper and Tag5Ink are now translated/rotated in their plane to match the visible portion in the saved episode-6 head image, retaining tag dimensions. Camera and other tags were not changed. This is a visual alignment under the current camera calibration; use the corrected USD as current geometry rather than archived tag_layout.json coordinates for tag 5. Fit and original room snapshot are stored in episode_006/physics/tag5_aligned.

## Episode 10: 1.0 mm thickness, followed by pink-only material revision

At user request, the same procedural Task000901 plate now has 1.0 mm vertical material separation at all six radial stations. All five visual bands and 160 convex collision sectors were regenerated, retaining 160 mm flat diameter, 190 mm outer diameter, 24 mm rim height and 15 g mass. The 1.5 mm predecessor is preserved in the grouped replay episode_010/physics/input/before_thickness_1mm directory. Task000200 generator remains unchanged. This is not a switch to task_901_haeun's separately scanned 18 cm plate.

After episode 10's video was complete, only pastel_pink.usda diffuseColor changed from (1.0, 0.94, 0.96) to (0.96, 0.72, 0.86). Blue/yellow, geometry, physics, cameras and lighting were retained. Direct rendered before/candidate/after comparisons are in the grouped vision sim_camera_color/pink_plate_color directory. Earlier videos keep their original material appearance.

Episodes 7 increased-friction revision and 8–10 use an episode-local gripper material override 4.5 static / 4.0 dynamic; common robot defaults remain 3.0/2.7. Episode9 frame120 (8 s) confirms visible and collider interpenetration despite enabled collision: current contact fidelity is not validated. Higher friction and thinner geometry must not be interpreted as a collision-quality fix. See episode_009/physics/collision_audit.


## 2026-11-03 corrected plate diameters: 185 / 150 mm

User corrected the outside diameter to 185 mm and flat base diameter to 150 mm. The shared procedural visual and convex collision geometry now use these dimensions. Height 24 mm, surface-normal thickness 1 mm, mass 15 g, and the modeled 75-degree upper wall remain fixed. The tangent inner fillet radius becomes 13.516 mm. Existing color layers are preserved. Previous 190/160 mm replay outputs remain historical; new diagnosis results are under pick_plate/episode_000/physics/friction_4p0_3p5/speed_0p5/diam185_base150_diagnosis.

## 2026-11-03 initial layout fitted to Pick Plate episode 0

The plate-holder scene defaults now use blue XY (-0.455079127, -0.211888476), pink (-0.667974996, -0.350098671), yellow (-0.467716524, -0.444826649), and holder XYZ (-0.546405650, 0.156555335, 0.7002) m with yaw 91.043 degrees. These supersede the historical placement above. Fixed-camera fitting balances initial head/wrist center and peg-cap residuals; this is not a measured world pose or exact image alignment. The 0.5x replay retained 15 g, 1 mm, friction 4.0/3.5 and the original joint targets. Final placement is more upright but still perched high on the holder. Evidence is in diam185_base150_diagnosis/aligned_layout/README.md under the replay experiment.
