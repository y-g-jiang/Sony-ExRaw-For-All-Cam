"""Single-matrix DNG preview with Bradford adaptation; no display transform baked into output."""
import numpy as np

def _profile_values(profile, code, default):
    for tag in profile.dng_ifd0_tags():
        if tag[0] == code:
            v = np.asarray(tag[3], np.float64)
            if tag[1] in (5, 10):
                pairs = v.reshape(-1, 2)
                v = pairs[:, 0] / pairs[:, 1]
            return v
    return np.asarray(default, np.float64)


def thumbnail_matrix(profile):
    """Single-matrix fallback preview with as-shot chromatic adaptation.

    This small fallback thumbnail does not emulate Adobe's profiles/look tables.
    Adobe compression regenerates previews with its own full DNG renderer.
    Full camera profiles and AsShotNeutral are preserved in the DNG itself.
    """
    cm = np.asarray(profile.CM1, np.float64).reshape(3, 3)
    cc = _profile_values(profile, 50723, np.eye(3)).reshape(3, 3)
    ab = np.diag(_profile_values(profile, 50727, [1, 1, 1]))
    neutral = np.asarray(profile.ASN, np.float64)
    if neutral.shape != (3,) or not np.isfinite(neutral).all() or (neutral <= 0).any():
        raise ValueError('Invalid as-shot neutral for DNG preview')
    camera_to_xyz = np.linalg.inv(ab @ cc @ cm)
    shot_white = camera_to_xyz @ neutral
    if not np.isfinite(shot_white).all() or shot_white[1] <= 0:
        raise ValueError('Invalid DNG as-shot white point')
    shot_white /= shot_white[1]
    bradford = np.array([[.8951,.2664,-.1614],[-.7502,1.7135,.0367],[.0389,-.0685,1.0296]])
    d65 = np.array([.95047, 1., 1.08883])
    cones = bradford @ shot_white
    if (cones <= 0).any():
        raise ValueError('Invalid DNG preview white-point adaptation')
    adapt = np.linalg.inv(bradford) @ np.diag((bradford @ d65) / cones) @ bradford
    xyz_to_srgb = np.array([[3.2404542,-1.5371385,-.4985314],
                            [-.969266,1.8760108,.041556],
                            [.0556434,-.2040259,1.0572252]])
    return xyz_to_srgb @ adapt @ camera_to_xyz

