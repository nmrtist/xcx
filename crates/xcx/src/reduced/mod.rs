// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Reduced-variable layer: the map from raw spin densities/gradients to libxc's
//! reduced variables, plus shared math helpers. Everything here is generic over
//! `num_dual::DualNum` so the same code produces both values and derivatives.
//!
//! Provenance: ported-from-libxc (MPL-2.0), `ref/libxc/maple/util.mpl`.

pub(crate) mod consts;
pub(crate) mod vars;
