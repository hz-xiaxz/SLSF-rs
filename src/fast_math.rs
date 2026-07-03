use std::simd::Simd;

pub(crate) use sleef::f64::{cos_u10 as cos, exp_u10 as exp, sincos_u10 as sin_cos};

#[inline]
pub(crate) fn sin_cos_x4(values: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    let (sin_values, cos_values) = sleef::f64x::sincos_u10(Simd::<f64, 4>::from_array(values));
    (sin_values.to_array(), cos_values.to_array())
}

#[inline]
pub(crate) fn exp_x4(values: [f64; 4]) -> [f64; 4] {
    sleef::f64x::exp_u10(Simd::<f64, 4>::from_array(values)).to_array()
}

#[inline]
pub(crate) fn sin_cos_x8(values: [f64; 8]) -> ([f64; 8], [f64; 8]) {
    let (sin_values, cos_values) = sleef::f64x::sincos_u10(Simd::<f64, 8>::from_array(values));
    (sin_values.to_array(), cos_values.to_array())
}

#[inline]
pub(crate) fn exp_x8(values: [f64; 8]) -> [f64; 8] {
    sleef::f64x::exp_u10(Simd::<f64, 8>::from_array(values)).to_array()
}

#[inline]
pub(crate) fn has_avx512_f64_lanes() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}
