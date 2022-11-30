//! Making floating-point behave: ordering, equivalence, hashing, and
//! constraints for floating-point types.
//!
//! Decorum provides traits that describe types using floating-point
//! representations and provides proxy types that wrap primitive floating-point
//! types. Proxy types implement a total ordering and constraints on the sets of
//! values that they may represent.
//!
//! Decorum requires Rust 1.43.0 or higher.
//!
//! # Floating-Point Classes
//!
//! Traits, proxy types, and constraints are based on three classes or subsets
//! of floating-point values:
//!
//! | Set          | Trait        |
//! |--------------|--------------|
//! | real number  | [`Real`]     |
//! | infinity     | [`Infinite`] |
//! | not-a-number | [`Nan`]      |
//!
//! Primitive floating-point values directly expose IEEE-754 and therefore the
//! complete set of values (and traits). Proxy types implement traits that are
//! compatible with their constraints, so types that disallow `NaN`s do not
//! implement the `Nan` trait, for example.
//!
//! # Proxy Types
//!
//! Proxy types wrap primitive floating-point types and constrain the sets of
//! values that they can represent:
//!
//! | Type       | Aliases      | Trait Implementations                      | Disallowed Values     |
//! |------------|--------------|--------------------------------------------|-----------------------|
//! | [`Total`]  |              | `Encoding + Real + Infinite + Nan + Float` |                       |
//! | [`NotNan`] | `N32`, `N64` | `Encoding + Real + Infinite`               | `NaN`                 |
//! | [`Finite`] | `R32`, `R64` | `Encoding + Real`                          | `NaN`, `-INF`, `+INF` |
//!
//! The [`NotNan`] and [`Finite`] types disallow values that represent `NaN`,
//! $\infin$, and $-\infin$. **Operations that emit values that violate these
//! constraints will panic**. The [`Total`] type applies no constraints and
//! exposes all classes of floating-point values.
//!
//! # Total Ordering
//!
//! The following total ordering is implemented by all proxy types and is
//! provided by traits in the [`cmp`] module:
//!
//! $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
//!
//! Note that all zero and `NaN` representations are considered equivalent. See
//! the [`cmp`] module documentation for more details.
//!
//! # Equivalence
//!
//! Floating-point `NaN`s have numerous representations and are incomparable.
//! Decorum considers all `NaN` representations equal to all other `NaN`
//! representations and any and all `NaN` representations are unequal to
//! non-`NaN` values.
//!
//! See the [`cmp`] module documentation for more details.
//!
//! [`cmp`]: crate::cmp
//! [`Finite`]: crate::Finite
//! [`Infinite`]: crate::Infinite
//! [`Nan`]: crate::Nan
//! [`NotNan`]: crate::NotNan
//! [`Real`]: crate::Real
//! [`Total`]: crate::Total

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum-favicon.ico"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/olson-sean-k/decorum/master/doc/decorum.svg?sanitize=true"
)]
#![no_std]
#![cfg_attr(all(nightly, feature = "unstable"), feature(try_trait_v2))]

#[cfg(feature = "std")]
extern crate std;

pub mod cmp;
mod constraint;
mod error;
pub mod hash;
mod proxy;

use core::mem;
use core::num::FpCategory;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num_traits::{PrimInt, Unsigned};

#[cfg(not(feature = "std"))]
pub(crate) use num_traits::float::FloatCore as ForeignFloat;
#[cfg(feature = "std")]
pub(crate) use num_traits::real::Real as ForeignReal;
#[cfg(feature = "std")]
pub(crate) use num_traits::Float as ForeignFloat;

use crate::cmp::{IntrinsicOrd, UndefinedError};
use crate::constraint::{Constraint, FiniteConstraint, NotNanConstraint, UnitConstraint};
use crate::proxy::{ErrorOf, ExpressionOf};

pub use crate::constraint::ConstraintViolation;
pub use crate::error::{Assert, Expression, TryExpression, TryOption, TryResult};
pub use crate::proxy::Proxy;

pub use Expression::{Defined, Undefined};

/// Floating-point representation with total ordering.
pub type Total<T> = Proxy<T, UnitConstraint>;

/// Floating-point representation that cannot be `NaN`.
///
/// If an operation emits `NaN`, then a panic will occur. Like [`Total`], this
/// type implements a total ordering.
///
/// [`Total`]: crate::Total
pub type NotNan<T, M = Assert> = Proxy<T, NotNanConstraint<M>>;

/// 32-bit floating-point representation that cannot be `NaN`.
pub type N32 = NotNan<f32>;
/// 64-bit floating-point representation that cannot be `NaN`.
pub type N64 = NotNan<f64>;

/// Floating-point representation that must be a real number.
///
/// If an operation emits `NaN` or infinities, then a panic will occur. Like
/// [`Total`], this type implements a total ordering.
///
/// [`Total`]: crate::Total
pub type Finite<T, M = Assert> = Proxy<T, FiniteConstraint<M>>;

/// 32-bit floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if
/// "F" were used, then this name would be very similar to `f32`.
pub type R32 = Finite<f32>;
/// 64-bit floating-point representation that must be a real number.
///
/// The prefix "R" for _real_ is used instead of "F" for _finite_, because if
/// "F" were used, then this name would be very similar to `f64`.
pub type R64 = Finite<f64>;

// TODO: Inverse the relationship between `Encoding` and `ToCanonicalBits` such
//       that `Encoding` requires `ToCanonicalBits`.
/// Converts floating-point values into a canonicalized form.
pub trait ToCanonicalBits: Encoding {
    type Bits: PrimInt + Unsigned;

    /// Conversion to a canonical representation.
    ///
    /// Unlike the `to_bits` function provided by `f32` and `f64`, this function
    /// collapses representations for real numbers, infinities, and `NaN`s into
    /// a canonical form such that every semantic value has a unique
    /// representation as canonical bits.
    fn to_canonical_bits(self) -> Self::Bits;
}

// TODO: Implement this differently for differently sized types.
impl<T> ToCanonicalBits for T
where
    T: Encoding + Nan + Primitive,
{
    type Bits = u64;

    fn to_canonical_bits(self) -> Self::Bits {
        const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
        const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;
        const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffff;

        const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
        const CANONICAL_ZERO_BITS: u64 = 0x0;

        if self.is_nan() {
            CANONICAL_NAN_BITS
        }
        else {
            let (mantissa, exponent, sign) = self.integer_decode();
            if mantissa == 0 {
                CANONICAL_ZERO_BITS
            }
            else {
                let exponent = u64::from(unsafe { mem::transmute::<i16, u16>(exponent) });
                let sign = if sign > 0 { 1u64 } else { 0u64 };
                (mantissa & MANTISSA_MASK)
                    | ((exponent << 52) & EXPONENT_MASK)
                    | ((sign << 63) & SIGN_MASK)
            }
        }
    }
}

/// Floating-point representations that expose infinities.
pub trait Infinite: Copy {
    const INFINITY: Self;
    const NEG_INFINITY: Self;

    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
}

impl<T, P> Infinite for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Copy,
    Proxy<T, P>: Infinite,
    T: Float + Primitive,
    P: Constraint<ErrorMode = TryExpression>,
{
    const INFINITY: Self = Defined(Infinite::INFINITY);
    const NEG_INFINITY: Self = Defined(Infinite::NEG_INFINITY);

    fn is_infinite(self) -> bool {
        match self {
            Defined(defined) => defined.is_infinite(),
            _ => false,
        }
    }

    fn is_finite(self) -> bool {
        match self {
            Defined(defined) => defined.is_finite(),
            _ => false,
        }
    }
}

/// Floating-point representations that expose `NaN`s.
pub trait Nan: Copy {
    /// A representation of `NaN`.
    ///
    /// For primitive floating-point types, `NaN` is incomparable. Therefore,
    /// prefer the `is_nan` predicate over direct comparisons with `NaN`.
    const NAN: Self;

    fn is_nan(self) -> bool;
}

/// Floating-point encoding.
///
/// Provides values and operations that describe the encoding of an IEEE-754
/// floating-point value. Infinities and `NaN`s are described by the `Infinite`
/// and `NaN` sub-traits.
pub trait Encoding: Copy {
    const MAX_FINITE: Self;
    const MIN_FINITE: Self;
    const MIN_POSITIVE_NORMAL: Self;
    const EPSILON: Self;

    fn classify(self) -> FpCategory;
    fn is_normal(self) -> bool;

    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;

    fn integer_decode(self) -> (u64, i16, i8);
}

impl Encoding for f32 {
    const MAX_FINITE: Self = f32::MAX;
    const MIN_FINITE: Self = f32::MIN;
    const MIN_POSITIVE_NORMAL: Self = f32::MIN_POSITIVE;
    const EPSILON: Self = f32::EPSILON;

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn is_sign_positive(self) -> bool {
        Self::is_sign_positive(self)
    }

    fn is_sign_negative(self) -> bool {
        Self::is_sign_negative(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
        let exponent: i16 = ((bits >> 23) & 0xff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x7f_ffff) << 1
        }
        else {
            (bits & 0x7f_ffff) | 0x80_0000
        };
        (mantissa as u64, exponent - (127 + 23), sign)
    }
}

impl Encoding for f64 {
    const MAX_FINITE: Self = f64::MAX;
    const MIN_FINITE: Self = f64::MIN;
    const MIN_POSITIVE_NORMAL: Self = f64::MIN_POSITIVE;
    const EPSILON: Self = f64::EPSILON;

    fn classify(self) -> FpCategory {
        self.classify()
    }

    fn is_normal(self) -> bool {
        self.is_normal()
    }

    fn is_sign_positive(self) -> bool {
        Self::is_sign_positive(self)
    }

    fn is_sign_negative(self) -> bool {
        Self::is_sign_negative(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits = self.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xf_ffff_ffff_ffff) << 1
        }
        else {
            (bits & 0xf_ffff_ffff_ffff) | 0x10_0000_0000_0000
        };
        (mantissa, exponent - (1023 + 52), sign)
    }
}

// TODO: Bounds on the following traits were removed, because they require
//       `NumOps` and that requires operation outputs to be closed:
//
//       - `Num`
//       - `Signed`
/// Types that can represent real numbers.
///
/// Provides values and operations that generally apply to real numbers. As
/// such, this trait is implemented by types using floating-point
/// representations, but this trait is a general numeric trait and can be
/// implemented by other numeric types as well.
///
/// Some members of this trait depend on the standard library and the `std`
/// feature.
pub trait Real:
    Add<Output = Self::Branch>
    + Copy
    + Div<Output = Self::Branch>
    + IntrinsicOrd // TODO: ???
    + Mul<Output = Self::Branch>
    + Neg<Output = Self>
    + PartialEq
    + PartialOrd
    + Rem<Output = Self::Branch>
    + Sub<Output = Self::Branch>
{
    type Branch;

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
    fn recip(self) -> Self::Branch; // Undefined or infinity.

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Branch; // Undefined or infinity.

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn cbrt(self) -> Self;
    #[cfg(feature = "std")]
    fn exp(self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self;
    #[cfg(feature = "std")]
    fn ln(self) -> Self;
    #[cfg(feature = "std")]
    fn log2(self) -> Self;
    #[cfg(feature = "std")]
    fn log10(self) -> Self;
    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self;

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn to_radians(self) -> Self;
    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Branch; // Overflow.
    #[cfg(feature = "std")]
    fn sin(self) -> Self;
    #[cfg(feature = "std")]
    fn cos(self) -> Self;
    #[cfg(feature = "std")]
    fn tan(self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn asin(self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn acos(self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn atan(self) -> Self;
    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self;
    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self);
    #[cfg(feature = "std")]
    fn sinh(self) -> Self;
    #[cfg(feature = "std")]
    fn cosh(self) -> Self;
    #[cfg(feature = "std")]
    fn tanh(self) -> Self;
    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Branch; // Undefined or infinity.
    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Branch; // Undefined or infinity.
}

impl<T, P> Real for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Copy + UndefinedError,
    T: Float + Primitive,
    P: Constraint<ErrorMode = TryExpression>,
{
    type Branch = Self;

    const ZERO: Self = Defined(Real::ZERO);
    const ONE: Self = Defined(Real::ONE);
    const E: Self = Defined(Real::E);
    const PI: Self = Defined(Real::PI);
    const FRAC_1_PI: Self = Defined(Real::FRAC_1_PI);
    const FRAC_2_PI: Self = Defined(Real::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = Defined(Real::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = Defined(Real::FRAC_PI_2);
    const FRAC_PI_3: Self = Defined(Real::FRAC_PI_3);
    const FRAC_PI_4: Self = Defined(Real::FRAC_PI_4);
    const FRAC_PI_6: Self = Defined(Real::FRAC_PI_6);
    const FRAC_PI_8: Self = Defined(Real::FRAC_PI_8);
    const SQRT_2: Self = Defined(Real::SQRT_2);
    const FRAC_1_SQRT_2: Self = Defined(Real::FRAC_1_SQRT_2);
    const LN_2: Self = Defined(Real::LN_2);
    const LN_10: Self = Defined(Real::LN_10);
    const LOG2_E: Self = Defined(Real::LOG2_E);
    const LOG10_E: Self = Defined(Real::LOG10_E);

    fn is_zero(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_zero(),
            _ => false,
        }
    }

    fn is_one(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_one(),
            _ => false,
        }
    }

    fn is_positive(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_positive(),
            _ => false,
        }
    }

    fn is_negative(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_negative(),
            _ => false,
        }
    }

    #[cfg(feature = "std")]
    fn abs(self) -> Self {
        self.map(Real::abs)
    }

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        self.map(Real::signum)
    }

    fn floor(self) -> Self {
        self.map(Real::floor)
    }

    fn ceil(self) -> Self {
        self.map(Real::ceil)
    }

    fn round(self) -> Self {
        self.map(Real::round)
    }

    fn trunc(self) -> Self {
        self.map(Real::trunc)
    }

    fn fract(self) -> Self {
        self.map(Real::fract)
    }

    fn recip(self) -> Self::Branch {
        self.and_then(Real::recip)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self::Branch {
        Real::mul_add(expression!(self), expression!(a), expression!(b))
    }

    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Branch {
        Real::div_euclid(expression!(self), expression!(n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Branch {
        Real::rem_euclid(expression!(self), expression!(n))
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Branch {
        self.and_then(|defined| Real::powi(defined, n))
    }

    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self::Branch {
        Real::powf(expression!(self), expression!(n))
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Branch {
        self.and_then(Real::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map(Real::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self::Branch {
        self.and_then(Real::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Branch {
        self.and_then(Real::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Branch {
        self.and_then(Real::exp_m1)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self {
        Defined(Real::log(expression!(self), expression!(base)))
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        self.map(Real::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self {
        self.map(Real::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self {
        self.map(Real::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self {
        self.map(Real::ln_1p)
    }

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Branch {
        self.and_then(Real::to_degrees)
    }

    #[cfg(feature = "std")]
    fn to_radians(self) -> Self {
        self.map(Real::to_radians)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Branch {
        Real::hypot(expression!(self), expression!(other))
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map(Real::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map(Real::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self::Branch {
        self.and_then(Real::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self::Branch {
        self.and_then(Real::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self::Branch {
        self.and_then(Real::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map(Real::atan)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self {
        Defined(Real::atan2(expression!(self), expression!(other)))
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        match self {
            Defined(defined) => {
                let (sin, cos) = defined.sin_cos();
                (Defined(sin), Defined(cos))
            }
            _ => (self, self),
        }
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map(Real::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map(Real::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map(Real::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Branch {
        self.and_then(Real::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Branch {
        self.and_then(Real::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Branch {
        self.and_then(Real::atanh)
    }
}

pub trait IntrinsicReal: IntrinsicOrd + Real<Branch = Self> {}

impl<T> IntrinsicReal for T where T: IntrinsicOrd + Real<Branch = T> {}

pub trait ClosedReal: Real<Branch = <Self as ClosedReal>::ClosedBranch> {
    type ClosedBranch: Real;
}

impl<T> ClosedReal for T
where
    T: Real,
    T::Branch: Real,
{
    type ClosedBranch = T::Branch;
}

/// Floating-point representations.
///
/// Types that implement this trait are represented using IEEE-754 encoding and
/// expose the details of that encoding, including infinities, `NaN`, and
/// operations on real numbers. This trait is implemented by primitive
/// floating-point types and the `Total` proxy type.
pub trait Float: Encoding + Infinite + IntrinsicOrd + Nan + Real<Branch = Self> {}

impl<T> Float for T where T: Encoding + Infinite + IntrinsicOrd + Nan + Real<Branch = T> {}

/// Primitive floating-point types.
pub trait Primitive {}

/// Implements floating-point traits for primitive types.
macro_rules! impl_primitive {
    (primitive => $t:ident) => {
        impl Infinite for $t {
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;

            fn is_infinite(self) -> bool {
                self.is_infinite()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }

        impl Nan for $t {
            const NAN: Self = <$t>::NAN;

            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }

        impl Primitive for $t {}

        impl Real for $t {
            type Branch = $t;

            // TODO: The propagation from a constant in a module requires that
            //       this macro accept an `ident` token rather than a `ty`
            //       token. Use `ty` if these constants become associated
            //       constants of the primitive types.
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const E: Self = core::$t::consts::E;
            const PI: Self = core::$t::consts::PI;
            const FRAC_1_PI: Self = core::$t::consts::FRAC_1_PI;
            const FRAC_2_PI: Self = core::$t::consts::FRAC_2_PI;
            const FRAC_2_SQRT_PI: Self = core::$t::consts::FRAC_2_SQRT_PI;
            const FRAC_PI_2: Self = core::$t::consts::FRAC_PI_2;
            const FRAC_PI_3: Self = core::$t::consts::FRAC_PI_3;
            const FRAC_PI_4: Self = core::$t::consts::FRAC_PI_4;
            const FRAC_PI_6: Self = core::$t::consts::FRAC_PI_6;
            const FRAC_PI_8: Self = core::$t::consts::FRAC_PI_8;
            const SQRT_2: Self = core::$t::consts::SQRT_2;
            const FRAC_1_SQRT_2: Self = core::$t::consts::FRAC_1_SQRT_2;
            const LN_2: Self = core::$t::consts::LN_2;
            const LN_10: Self = core::$t::consts::LN_10;
            const LOG2_E: Self = core::$t::consts::LOG2_E;
            const LOG10_E: Self = core::$t::consts::LOG10_E;

            fn is_zero(self) -> bool {
                self == Self::ZERO
            }

            fn is_one(self) -> bool {
                self == Self::ONE
            }

            fn is_positive(self) -> bool {
                <$t>::is_sign_positive(self)
            }

            fn is_negative(self) -> bool {
                <$t>::is_sign_negative(self)
            }

            #[cfg(feature = "std")]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            #[cfg(feature = "std")]
            fn signum(self) -> Self {
                <$t>::signum(self)
            }

            fn floor(self) -> Self {
                <$t>::floor(self)
            }

            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }

            fn round(self) -> Self {
                <$t>::round(self)
            }

            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }

            fn fract(self) -> Self {
                <$t>::fract(self)
            }

            fn recip(self) -> Self::Branch {
                <$t>::recip(self)
            }

            #[cfg(feature = "std")]
            fn mul_add(self, a: Self, b: Self) -> Self::Branch {
                <$t>::mul_add(self, a, b)
            }

            #[cfg(feature = "std")]
            fn div_euclid(self, n: Self) -> Self::Branch {
                <$t>::div_euclid(self, n)
            }

            #[cfg(feature = "std")]
            fn rem_euclid(self, n: Self) -> Self::Branch {
                <$t>::rem_euclid(self, n)
            }

            #[cfg(feature = "std")]
            fn powi(self, n: i32) -> Self::Branch {
                <$t>::powi(self, n)
            }

            #[cfg(feature = "std")]
            fn powf(self, n: Self) -> Self::Branch {
                <$t>::powf(self, n)
            }

            #[cfg(feature = "std")]
            fn sqrt(self) -> Self::Branch {
                <$t>::sqrt(self)
            }

            #[cfg(feature = "std")]
            fn cbrt(self) -> Self {
                <$t>::cbrt(self)
            }

            #[cfg(feature = "std")]
            fn exp(self) -> Self::Branch {
                <$t>::exp(self)
            }

            #[cfg(feature = "std")]
            fn exp2(self) -> Self::Branch {
                <$t>::exp2(self)
            }

            #[cfg(feature = "std")]
            fn exp_m1(self) -> Self::Branch {
                <$t>::exp_m1(self)
            }

            #[cfg(feature = "std")]
            fn log(self, base: Self) -> Self {
                <$t>::log(self, base)
            }

            #[cfg(feature = "std")]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            #[cfg(feature = "std")]
            fn log2(self) -> Self {
                <$t>::log2(self)
            }

            #[cfg(feature = "std")]
            fn log10(self) -> Self {
                <$t>::log10(self)
            }

            #[cfg(feature = "std")]
            fn ln_1p(self) -> Self {
                <$t>::ln_1p(self)
            }

            #[cfg(feature = "std")]
            fn to_degrees(self) -> Self::Branch {
                <$t>::to_degrees(self)
            }

            #[cfg(feature = "std")]
            fn to_radians(self) -> Self {
                <$t>::to_radians(self)
            }

            #[cfg(feature = "std")]
            fn hypot(self, other: Self) -> Self::Branch {
                <$t>::hypot(self, other)
            }

            #[cfg(feature = "std")]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }

            #[cfg(feature = "std")]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }

            #[cfg(feature = "std")]
            fn tan(self) -> Self::Branch {
                <$t>::tan(self)
            }

            #[cfg(feature = "std")]
            fn asin(self) -> Self::Branch {
                <$t>::asin(self)
            }

            #[cfg(feature = "std")]
            fn acos(self) -> Self::Branch {
                <$t>::acos(self)
            }

            #[cfg(feature = "std")]
            fn atan(self) -> Self {
                <$t>::atan(self)
            }

            #[cfg(feature = "std")]
            fn atan2(self, other: Self) -> Self {
                <$t>::atan2(self, other)
            }

            #[cfg(feature = "std")]
            fn sin_cos(self) -> (Self, Self) {
                <$t>::sin_cos(self)
            }

            #[cfg(feature = "std")]
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }

            #[cfg(feature = "std")]
            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }

            #[cfg(feature = "std")]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }

            #[cfg(feature = "std")]
            fn asinh(self) -> Self::Branch {
                <$t>::asinh(self)
            }

            #[cfg(feature = "std")]
            fn acosh(self) -> Self::Branch {
                <$t>::acosh(self)
            }

            #[cfg(feature = "std")]
            fn atanh(self) -> Self::Branch {
                <$t>::atanh(self)
            }
        }
    };
}
impl_primitive!(primitive => f32);
impl_primitive!(primitive => f64);
