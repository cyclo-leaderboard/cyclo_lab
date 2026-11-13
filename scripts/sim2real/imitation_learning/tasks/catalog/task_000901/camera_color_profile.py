"""Calibrated Task000901 display response for the three simulated cameras.

The profile is fitted from semantic material patches in the 2026-11-10 live
capture and the matching raw Isaac render.  Processing stays in linear RGB:
3x3 CCM, one monotone luminance curve, then a soft highlight shoulder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


PROFILES = {
    "head": {
        "matrix": (
            (1.17000000, -0.04350000, -0.08400000),
            (0.07761365, 1.01321929, -0.08064111),
            (0.01920000, 0.07580000, 0.83500000),
        ),
        "tone_scale": 1.220,
        "tone_gamma": 1.800,
        "highlight_knee": 0.760,
        "highlight_shoulder": 0.130,
        "shadow_end": 0.04121766,
        "shadow_lift_rgb": (0.0068, 0.0049, 0.0036),
        "shadow_detail_gain": 2.20,
        # Reduce the display-referred veil so the head image keeps clearer
        # wall/table separation while retaining the fitted warm white response.
        "white_veil": 0.100,
        "shadow_veil_protection": 0.20,
        "highlight_neutralization": 0.75,
        "highlight_neutral_start": 0.50,
        "highlight_neutral_chroma_end": 0.24,
        "highlight_cool_lift_rgb": (0.000, 0.003, 0.014),
        "white_response_lift": 0.12,
        "colored_chroma_gain": 1.18,
        "colored_chroma_start": 0.035,
        "colored_chroma_full": 0.12,
        "colored_shadow_bias": 0.25,
        "colored_shadow_lift": 0.80,
        # Camera-specific plate hues fitted against EP20 head reference.
        "blue_gate_start": 0.00,
        "blue_gate_full": 0.04,
        "blue_green_gate_start": -0.04,
        "blue_green_gate_full": 0.00,
        "blue_luma_start": 0.70,
        "blue_luma_full": 0.85,
        "blue_channel_gain": (1.0, 1.0, 1.0),
        "blue_channel_lift": (0.10, 0.80, 0.95),
        "pink_red_gate_start": 0.01,
        "pink_red_gate_full": 0.03,
        "pink_blue_gate_start": -0.15,
        "pink_blue_gate_full": -0.04,
        "pink_luma_start": 0.50,
        "pink_luma_full": 0.85,
        "pink_channel_lift": (0.70, 0.80, 0.82),
        "yellow_channel_lift": (0.00, 0.05, 0.35),
        # Head-only amber/brown response, fitted to the real wooden holder.
        "warm_material_gamma": 1.35,
        "warm_material_pivot": 0.88,
        "warm_material_gain_rgb": (0.92, 0.91, 0.86),
        "yellow_shadow_blue_gain": 0.60,
        "softening_strength": 0.50,
    },
    "wrist_left": {
        "matrix": (
            (1.03028101, -0.03046996, -0.11311871),
            (-0.01818536, 0.97389979, 0.03288073),
            (0.09847765, 0.05089439, 1.01725128),
        ),
        "tone_scale": 1.240,
        "tone_gamma": 2.000,
        "highlight_knee": 0.620,
        "highlight_shoulder": 1.000,
        "shadow_end": 0.040,
        "shadow_lift_rgb": (0.0002, 0.0033, 0.0030),
        # Retain the v36 framing and expose the authored fingertip grooves.
        "shadow_detail_gain": 6.0,
        # EP20/21 neutral tabletop patches: lift the brighter gray midtones
        # while preserving the darker rear edge and saturated plate response.
        "neutral_midtone_lift_rgb": (0.025, 0.025, 0.075),
        "neutral_midtone_luma_range": (0.35, 0.44, 0.55, 0.72),
        "neutral_midtone_chroma_range": (0.055, 0.120),
        "white_veil": 0.060,
        "shadow_veil_protection": 0.30,
        # Emphasize near-white highlights while retaining their gradients.
        "highlight_neutral_start": 0.35,
        "highlight_neutral_chroma_end": 0.40,
        "highlight_neutralization": 0.25,
        "white_response_lift": 0.90,
        "colored_chroma_gain": 1.35,
        "colored_chroma_start": 0.035,
        "colored_chroma_full": 0.18,
        # Lift shaded colored surfaces from their raw hue, avoiding the
        # blue-gray wall response and keeping white highlights graded.
        "colored_shadow_lift": 0.55,
        "colored_shadow_source_selective": True,
        "display_shadow_fill": 0.06,
        "cool_midtone_gain": 0.88,
        "colored_highlight_lift": 0.20,
        "colored_highlight_start": 0.24,
        # Raise red/green with blue highlights for a brighter near-white
        # response while preserving a positive slope in every channel.
        "blue_gate_start": 0.12,
        "blue_gate_full": 0.22,
        "blue_green_gate_start": 0.015,
        "blue_green_gate_full": 0.060,
        "blue_luma_start": 0.52,
        "blue_luma_full": 0.80,
        "blue_channel_gain": (0.735, 0.880, 1.000),
        "blue_channel_lift": (0.080, 0.040, 0.650),
        "pink_red_gate_start": 0.000,
        "pink_red_gate_full": 0.012,
        "pink_blue_gate_start": 0.015,
        "pink_blue_gate_full": 0.050,
        "pink_luma_start": 0.45,
        "pink_luma_full": 0.75,
        "pink_channel_lift": (0.250, 0.050, 0.980),
        "pink_channel_gain": (0.98, 0.93, 1.0),
        "pink_source_selective": True,
        "yellow_channel_lift": (0.20, 0.15, 0.00),
        "softening_strength": 0.45,
    },
    "wrist_right": {
        "matrix": (
            (1.00999980, -0.02308256, -0.03129511),
            (-0.01030913, 0.99530945, -0.00301799),
            (0.07267492, 0.11443265, 1.12204718),
        ),
        "tone_scale": 1.010,
        "tone_gamma": 1.350,
        "highlight_knee": 0.620,
        "highlight_shoulder": 1.000,
        "shadow_end": 0.055,
        "shadow_lift_rgb": (0.0078, 0.0137, 0.0241),
        # The right wrist needs stronger deep-shadow separation for the grooves.
        "shadow_detail_gain": 16.0,
        "white_veil": 0.020,
        "shadow_veil_protection": 0.30,
        # Match the stronger near-white response requested for both wrists.
        "highlight_neutral_start": 0.35,
        "highlight_neutral_chroma_end": 0.40,
        "highlight_neutralization": 0.25,
        "white_response_lift": 0.90,
        "warm_red_start": 0.015,
        "warm_red_full": 0.11,
        "warm_blue_tolerance": 0.07,
        "warm_tint_lift_rgb": (0.085, 0.018, 0.035),
        # Match the left wrist blue-to-white highlight response while
        # retaining the independently fitted warm midtone response.
        "blue_gate_start": 0.12,
        "blue_gate_full": 0.22,
        "blue_green_gate_start": 0.015,
        "blue_green_gate_full": 0.060,
        "blue_luma_start": 0.52,
        "blue_luma_full": 0.80,
        "blue_channel_gain": (0.860, 0.960, 1.000),
        # Keep response slopes positive instead of clipping bright plate areas.
        "blue_channel_lift": (0.160, 0.080, 0.750),
        "pink_red_gate_start": 0.000,
        "pink_red_gate_full": 0.012,
        "pink_blue_gate_start": 0.015,
        "pink_blue_gate_full": 0.050,
        "pink_luma_start": 0.45,
        "pink_luma_full": 0.75,
        "pink_channel_lift": (0.750, 0.600, 0.950),
        "pink_channel_gain": (0.99, 0.97, 1.0),
        "yellow_channel_lift": (0.25, 0.22, 0.08),
        "softening_strength": 0.25,
    },
}

_LUMA = (0.2126, 0.7152, 0.0722)


def _srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def _linear_to_srgb(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )


def apply_camera_profile(raw_rgb: torch.Tensor, camera: str) -> torch.Tensor:
    """Apply the fitted camera profile to an RGB uint8-like tensor."""
    profile = PROFILES[camera]
    signal = _srgb_to_linear((raw_rgb.to(torch.float32) / 255.0).clamp(0.0, 1.0))
    matrix = torch.as_tensor(profile["matrix"], device=signal.device, dtype=signal.dtype)
    corrected = torch.matmul(signal, matrix.transpose(0, 1)).clamp(0.0, 1.0)

    luma_weights = torch.as_tensor(_LUMA, device=signal.device, dtype=signal.dtype)
    old_luma = (corrected * luma_weights).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    new_luma = (profile["tone_scale"] * old_luma).clamp(0.0, 1.0).pow(profile["tone_gamma"])

    knee = profile["highlight_knee"]
    shoulder = profile["highlight_shoulder"]
    shoulder_t = ((new_luma - knee) / (1.0 - knee)).clamp(0.0, 1.0)
    compressed = torch.expm1(shoulder * shoulder_t) / torch.expm1(
        torch.as_tensor(shoulder, device=signal.device, dtype=signal.dtype)
    )
    rolled_luma = torch.where(
        new_luma > knee,
        knee + (1.0 - knee) * compressed,
        new_luma,
    )
    corrected = corrected * (rolled_luma / old_luma)
    # Real sensors retain a colored black level in deep shadow.  Apply the
    # measured camera-specific toe only below the configured linear-luma end;
    # midtones, plate colors, and highlights remain governed by the fitted
    # CCM/tone/shoulder stages above.
    shadow_end = profile["shadow_end"]
    shadow_weight = (1.0 - old_luma / shadow_end).clamp(0.0, 1.0).square()
    shadow_lift = torch.as_tensor(
        profile["shadow_lift_rgb"], device=signal.device, dtype=signal.dtype
    )
    corrected = corrected + shadow_weight * shadow_lift
    # The authored fingertip grooves are only a few code values above black in
    # the raw render. Restore their local signal slope after lifting the sensor
    # black level so the arrow pattern is not flattened into a solid silhouette.
    corrected = corrected + shadow_weight * profile["shadow_detail_gain"] * signal
    display_rgb = _linear_to_srgb(corrected)
    # A small display-referred veiling component models the washed, low-contrast
    # look of the live camera without changing scene lighting or material color.
    veil = profile["white_veil"]
    veil_weight = 1.0 - profile["shadow_veil_protection"] * shadow_weight
    display_rgb = display_rgb + veil * veil_weight * (1.0 - display_rgb)
    # Keep the warm response in shadows and midtones, while neutralizing the
    # ivory cast in bright whites. A small blue-only highlight lift preserves
    # the cool response visible on white surfaces in the real head camera.
    display_luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True)
    neutral_start = profile.get("highlight_neutral_start", 1.0)
    highlight_weight = (
        (display_luma - neutral_start) / max(1.0 - neutral_start, 1e-6)
    ).clamp(0.0, 1.0).square()
    display_chroma = display_rgb.amax(dim=-1, keepdim=True) - display_rgb.amin(
        dim=-1, keepdim=True
    )
    neutral_chroma_end = profile.get("highlight_neutral_chroma_end", 1.0)
    neutral_chroma_gate = (
        1.0 - display_chroma / max(neutral_chroma_end, 1e-6)
    ).clamp(0.0, 1.0).square()
    neutral_weight = highlight_weight * neutral_chroma_gate
    neutral_rgb = display_luma.expand_as(display_rgb)
    display_rgb = torch.lerp(
        display_rgb,
        neutral_rgb,
        neutral_weight * profile.get("highlight_neutralization", 0.0),
    )
    # Veiling haze lowers the color separation in plate highlights and their
    # shaded rims. Restore chroma only where color is already present; neutral
    # walls, table regions, tags, and white highlights remain unaffected.
    chroma_start = profile.get("colored_chroma_start", 1.0)
    chroma_full = profile.get("colored_chroma_full", 1.0)
    colored_gate = (
        (display_chroma - chroma_start) / max(chroma_full - chroma_start, 1e-6)
    ).clamp(0.0, 1.0)
    shadow_bias = 1.0 + profile.get("colored_shadow_bias", 0.0) * (
        1.0 - display_luma
    )
    colored_highlight_start = profile.get("colored_highlight_start", 1.0)
    colored_highlight_weight = colored_gate * (
        (display_luma - colored_highlight_start)
        / max(1.0 - colored_highlight_start, 1e-6)
    ).clamp(0.0, 1.0).square()
    display_rgb = (
        display_rgb
        + colored_highlight_weight
        * profile.get("colored_highlight_lift", 0.0)
        * display_rgb
        * (1.0 - display_rgb)
    ).clamp(0.0, 1.0)
    chroma_gain = 1.0 + (
        profile.get("colored_chroma_gain", 1.0) - 1.0
    ) * colored_gate * shadow_bias
    display_rgb = (
        display_luma + (display_rgb - display_luma) * chroma_gain
    ).clamp(0.0, 1.0)
    shadow_color_gate = colored_gate
    if profile.get("colored_shadow_source_selective", False):
        source_rgb = (raw_rgb.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        red, green, blue = source_rgb.split(1, dim=-1)
        warm_gate = ((red - green - 0.025) / 0.040).clamp(0.0, 1.0)
        yellow_gate = ((torch.minimum(red, green) - blue - 0.05) / 0.08).clamp(0.0, 1.0)
        blue_plate_gate = ((blue - red - 0.11) / 0.13).clamp(0.0, 1.0)
        shadow_color_gate = torch.maximum(torch.maximum(warm_gate, yellow_gate), blue_plate_gate)
    colored_shadow_weight = shadow_color_gate * (1.0 - display_luma).square()
    display_rgb = (
        display_rgb
        + colored_shadow_weight
        * profile.get("colored_shadow_lift", 0.0)
        * (1.0 - display_rgb)
    ).clamp(0.0, 1.0)
    red_over_green = display_rgb[..., 0:1] - display_rgb[..., 1:2]
    green_over_blue = display_rgb[..., 1:2] - display_rgb[..., 2:3]
    warm_gate = (
        (red_over_green - profile.get("warm_red_start", 1.0))
        / max(
            profile.get("warm_red_full", 1.0)
            - profile.get("warm_red_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    warm_gate = warm_gate * (
        (green_over_blue + profile.get("warm_blue_tolerance", 0.0)) / 0.12
    ).clamp(0.0, 1.0)
    warm_tint = torch.as_tensor(
        profile.get("warm_tint_lift_rgb", (0.0, 0.0, 0.0)),
        device=signal.device,
        dtype=signal.dtype,
    )
    display_rgb = (
        display_rgb + warm_gate * warm_tint * (1.0 - display_rgb)
    ).clamp(0.0, 1.0)
    cool_lift = torch.as_tensor(
        profile.get("highlight_cool_lift_rgb", (0.0, 0.0, 0.0)),
        device=signal.device,
        dtype=signal.dtype,
    )
    display_rgb = (display_rgb + neutral_weight * cool_lift).clamp(0.0, 1.0)
    display_rgb = (
        display_rgb
        + neutral_weight
        * profile.get("white_response_lift", 0.0)
        * (1.0 - display_rgb)
    ).clamp(0.0, 1.0)
    # The two D405 wrist cameras retain substantially more spectral separation
    # in glossy plate highlights than the head ZED.  Gate by hue as well as
    # luma so neutral walls, white tags, wood, and the table response stay
    # unchanged.  The affine channel response preserves highlight gradients
    # instead of painting a constant target color onto the plate.
    blue_over_red = display_rgb[..., 2:3] - display_rgb[..., 0:1]
    blue_over_green = display_rgb[..., 2:3] - display_rgb[..., 1:2]
    blue_gate = (
        (blue_over_red - profile.get("blue_gate_start", 1.0))
        / max(
            profile.get("blue_gate_full", 1.0)
            - profile.get("blue_gate_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    blue_gate = blue_gate * (
        (blue_over_green - profile.get("blue_green_gate_start", 1.0))
        / max(
            profile.get("blue_green_gate_full", 1.0)
            - profile.get("blue_green_gate_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    blue_gate = blue_gate * (
        (display_luma - profile.get("blue_luma_start", 1.0))
        / max(
            profile.get("blue_luma_full", 1.0)
            - profile.get("blue_luma_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    blue_gain = torch.as_tensor(
        profile.get("blue_channel_gain", (1.0, 1.0, 1.0)),
        device=signal.device,
        dtype=signal.dtype,
    )
    blue_lift = torch.as_tensor(
        profile.get("blue_channel_lift", (0.0, 0.0, 0.0)),
        device=signal.device,
        dtype=signal.dtype,
    )
    blue_response = display_rgb * blue_gain + blue_lift * (1.0 - display_rgb)
    display_rgb = torch.lerp(display_rgb, blue_response, blue_gate).clamp(0.0, 1.0)

    red_over_green = display_rgb[..., 0:1] - display_rgb[..., 1:2]
    blue_over_green = display_rgb[..., 2:3] - display_rgb[..., 1:2]
    pink_gate = (
        (red_over_green - profile.get("pink_red_gate_start", 1.0))
        / max(
            profile.get("pink_red_gate_full", 1.0)
            - profile.get("pink_red_gate_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    pink_gate = pink_gate * (
        (blue_over_green - profile.get("pink_blue_gate_start", 1.0))
        / max(
            profile.get("pink_blue_gate_full", 1.0)
            - profile.get("pink_blue_gate_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    pink_gate = pink_gate * (
        (display_luma - profile.get("pink_luma_start", 1.0))
        / max(
            profile.get("pink_luma_full", 1.0)
            - profile.get("pink_luma_start", 1.0),
            1e-6,
        )
    ).clamp(0.0, 1.0)
    if profile.get("pink_source_selective", False):
        # Preserve pink identification when the white response has already
        # neutralized its highlights. Select hue from the original sensor RGB.
        source_rgb = (raw_rgb.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        source_r, source_g, source_b = source_rgb.split(1, dim=-1)
        source_pink = ((source_r - source_g - 0.015) / 0.030).clamp(0.0, 1.0)
        source_pink = source_pink * ((source_b - source_g + 0.08) / 0.10).clamp(0.0, 1.0)
        pink_brightness = ((display_luma - profile["pink_luma_start"]) / (profile["pink_luma_full"] - profile["pink_luma_start"])).clamp(0.0, 1.0)
        pink_gate = torch.maximum(pink_gate, source_pink * pink_brightness)
    pink_lift = torch.as_tensor(
        profile.get("pink_channel_lift", (0.0, 0.0, 0.0)),
        device=signal.device,
        dtype=signal.dtype,
    )
    pink_gain = torch.as_tensor(
        profile.get("pink_channel_gain", (1.0, 1.0, 1.0)),
        device=signal.device, dtype=signal.dtype,
    )
    pink_response = display_rgb * pink_gain + pink_lift * (1.0 - display_rgb)
    display_rgb = torch.lerp(display_rgb, pink_response, pink_gate).clamp(0.0, 1.0)
    if "yellow_channel_lift" in profile:
        # Correct the bright yellow response independently of near-white pink
        # and cyan; no image coordinates or object masks are used.
        yellow_delta = torch.minimum(display_rgb[..., 0:1], display_rgb[..., 1:2]) - display_rgb[..., 2:3]
        yellow_luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True)
        yellow_weight = ((yellow_delta - 0.10) / 0.15).clamp(0.0, 1.0)
        yellow_weight = yellow_weight * ((yellow_luma - 0.65) / 0.20).clamp(0.0, 1.0)
        yellow_lift = torch.as_tensor(profile["yellow_channel_lift"], device=signal.device, dtype=signal.dtype)
        display_rgb = (display_rgb + yellow_weight * yellow_lift * (1.0 - display_rgb)).clamp(0.0, 1.0)
    if "warm_material_gamma" in profile:
        # Identify amber/brown from raw hue, so brighter pink/yellow/blue
        # plates retain their independently fitted highlight responses.
        source_rgb = (raw_rgb.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        source_r, source_g, source_b = source_rgb.split(1, dim=-1)
        amber_r = ((source_r - source_g - 0.025) / 0.050).clamp(0.0, 1.0)
        amber_b = ((source_g - source_b - 0.025) / 0.050).clamp(0.0, 1.0)
        amber_weight = amber_r * amber_b
        amber_luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        amber_contrast = (amber_luma / profile["warm_material_pivot"]).pow(profile["warm_material_gamma"] - 1.0)
        amber_gain = torch.as_tensor(profile["warm_material_gain_rgb"], device=signal.device, dtype=signal.dtype)
        amber_response = display_rgb * amber_contrast * amber_gain
        display_rgb = torch.lerp(display_rgb, amber_response, amber_weight).clamp(0.0, 1.0)
    if "yellow_shadow_blue_gain" in profile:
        # Head-only yellow separation in shaded rims. Detect the original hue
        # and protect bright highlights, neutral tags, cyan and amber wood.
        source = (raw_rgb.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        r, g, b = source.split(1, dim=-1)
        yellow = ((torch.minimum(r, g) - b - 0.08) / 0.12).clamp(0.0, 1.0)
        yellow = yellow * ((0.06 - (r - g).abs()) / 0.03).clamp(0.0, 1.0)
        luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True)
        shade = ((0.96 - luma) / 0.18).clamp(0.0, 1.0)
        factor = 1.0 + yellow * shade * (profile["yellow_shadow_blue_gain"] - 1.0)
        display_rgb = torch.cat((display_rgb[..., :2], display_rgb[..., 2:3] * factor), dim=-1)
    if "display_shadow_fill" in profile:
        fill_luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True)
        fill_gate = ((fill_luma - 0.10) / 0.12).clamp(0.0, 1.0)
        fill_gate = fill_gate * ((0.45 - fill_luma) / 0.23).clamp(0.0, 1.0)
        display_rgb = display_rgb + profile["display_shadow_fill"] * fill_gate * (1.0 - display_rgb)
    if "cool_midtone_gain" in profile:
        # Correct the blue-gray midtone response without darkening neutral
        # tabletop tones or strongly colored plate highlights.
        cool_delta = display_rgb[..., 2:3] - display_rgb[..., 0:1]
        cool_green = display_rgb[..., 1:2] - display_rgb[..., 0:1]
        cool_luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True)
        cool_weight = ((cool_delta - 0.055) / 0.055).clamp(0.0, 1.0)
        cool_weight = cool_weight * ((0.24 - cool_delta) / 0.08).clamp(0.0, 1.0)
        cool_weight = cool_weight * (cool_green / 0.035).clamp(0.0, 1.0)
        cool_weight = cool_weight * ((cool_luma - 0.16) / 0.12).clamp(0.0, 1.0)
        cool_weight = cool_weight * ((0.70 - cool_luma) / 0.18).clamp(0.0, 1.0)
        display_rgb = display_rgb * (1.0 + cool_weight * (profile["cool_midtone_gain"] - 1.0))
    if "neutral_midtone_lift_rgb" in profile:
        # A smooth color/luminance response, independent of image position.
        # Limit it to near-neutral midtones; do not lift black grippers or
        # colored plate highlights together with the gray tabletop.
        mid_luma = (display_rgb * luma_weights).sum(dim=-1, keepdim=True)
        mid_chroma = display_rgb.amax(dim=-1, keepdim=True) - display_rgb.amin(
            dim=-1, keepdim=True
        )
        lo, full_lo, full_hi, hi = profile["neutral_midtone_luma_range"]
        chroma_lo, chroma_hi = profile["neutral_midtone_chroma_range"]
        rise = ((mid_luma - lo) / (full_lo - lo)).clamp(0.0, 1.0)
        fall = ((hi - mid_luma) / (hi - full_hi)).clamp(0.0, 1.0)
        neutral = ((chroma_hi - mid_chroma) / (chroma_hi - chroma_lo)).clamp(0.0, 1.0)
        weight = (rise.square() * (3.0 - 2.0 * rise)) * (
            fall.square() * (3.0 - 2.0 * fall)
        ) * (neutral.square() * (3.0 - 2.0 * neutral))
        mid_lift = torch.as_tensor(
            profile["neutral_midtone_lift_rgb"], device=signal.device, dtype=signal.dtype
        )
        display_rgb = (display_rgb + weight * mid_lift).clamp(0.0, 1.0)
    # Real camera MTF is softer than the renderer. Blend a small 3x3 sensor
    # footprint instead of modifying geometry, lighting, or material textures.
    channels_first = display_rgb.permute(0, 3, 1, 2)
    softened = F.avg_pool2d(F.pad(channels_first, (1, 1, 1, 1), mode="replicate"), 3, 1)
    channels_first = torch.lerp(
        channels_first, softened, profile["softening_strength"]
    )
    display_rgb = channels_first.permute(0, 2, 3, 1)
    return 255.0 * display_rgb
