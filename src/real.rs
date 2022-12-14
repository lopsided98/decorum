use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::cmp::IntrinsicOrd;
use crate::Infinite;

pub trait Codomain {
    type Superset;
}

pub trait Endofunction: Codomain<Superset = Self> {}

impl<T> Endofunction for T where T: Codomain<Superset = T> {}

pub trait UnaryReal:
    Codomain + IntrinsicOrd + Neg<Output = Self> + PartialEq + PartialOrd + Sized
{
    const ZERO: Self;
    const ONE: Self;
    const E: Self;
    const PI: Self;
    const FRAC_1_PI: Self;
    const FRAC_2_PI: Self;
    const FRAC_2_SQRT_PI: Self;
    const FRAC_PI_2: Self;
    const FRAC_PI_3: Self;
    const FRAC_PI_4: Self;
    const FRAC_PI_6: Self;
    const FRAC_PI_8: Self;
    const SQRT_2: Self;
    const FRAC_1_SQRT_2: Self;
    const LN_2: Self;
    const LN_10: Self;
    const LOG2_E: Self;
    const LOG10_E: Self;

    fn is_zero(self) -> bool;
    fn is_one(self) -> bool;

    fn is_positive(self) -> bool;
    fn is_negative(self) -> bool;
    #[cfg(feature = "std")]
    fn abs(self) -> Self;
    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        if self.is_positive() {
            Self::ONE
        }
        else {
            -Self::ONE
        }
    }

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn recip(self) -> Self::Superset; // Undefined or infinity.

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Superset; // Overflow, undefined, or infinity.
    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn cbrt(self) -> Self;
    #[cfg(feature = "std")]
    fn exp(self) -> Self::Superset; // Overflow.
    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Superset; // Overflow.
    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Superset; // Overflow.
    #[cfg(feature = "std")]
    fn ln(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn log2(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn log10(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Superset; // Undefined or infinity.

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Superset; // Overflow.
    #[cfg(feature = "std")]
    fn to_radians(self) -> Self;
    #[cfg(feature = "std")]
    fn sin(self) -> Self;
    #[cfg(feature = "std")]
    fn cos(self) -> Self;
    #[cfg(feature = "std")]
    fn tan(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn asin(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn acos(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn atan(self) -> Self;
    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self);
    #[cfg(feature = "std")]
    fn sinh(self) -> Self;
    #[cfg(feature = "std")]
    fn cosh(self) -> Self;
    #[cfg(feature = "std")]
    fn tanh(self) -> Self;
    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Superset; // Undefined or infinity.
}

// NOTE: Because `T` is not constrained, it isn't possible for functions that
//       always map reals to reals to express their output as `Self`. The `T`
//       input may not be real and that may result in a non-real output.
pub trait BinaryReal<T = Self>:
    Add<T, Output = Self::Superset>
    + Div<T, Output = Self::Superset>
    + Mul<T, Output = Self::Superset>
    + Rem<T, Output = Self::Superset>
    + Sub<T, Output = Self::Superset>
    + UnaryReal
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Superset; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Superset; // Undefined or infinity.

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Superset; // Overflow, undefined, or infinity.
    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Superset; // Undefined or infinity.

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Superset; // Overflow.
    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Superset;
}

pub trait Real: BinaryReal<Self> {}

impl<T> Real for T where T: BinaryReal<T> {}

pub trait ExtendedReal: Infinite + Real {}

impl<T> ExtendedReal for T where T: Infinite + Real {}
