//! Proxy types that wrap primitive floating-point types and apply constraints
//! and a total ordering.

#[cfg(feature = "approx")]
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use core::cmp::Ordering;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display, Formatter, LowerExp, UpperExp};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::mem;
use core::num::FpCategory;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::str::FromStr;
use num_traits::{
    Bounded, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero,
};
#[cfg(feature = "serialize-serde")]
use serde_derive::{Deserialize, Serialize};

use crate::cmp::{self, FloatEq, FloatOrd, IntrinsicOrd};
use crate::constraint::{
    Constraint, ConstraintViolation, ExpectConstrained, InfinitySet, Member, NanSet, SubsetOf,
    SupersetOf,
};
use crate::divergence::{Divergence, Expression, NonResidual, TryExpression};
use crate::hash::FloatHash;
#[cfg(feature = "std")]
use crate::ForeignReal;
use crate::{
    with_binary_operations, with_primitives, BinaryReal, Codomain, Encoding, Finite, Float,
    ForeignFloat, Infinite, Nan, NotNan, Primitive, ToCanonicalBits, Total, UnaryReal,
};

pub type BranchOf<P> = <DivergenceOf<P> as Divergence>::Branch<P, ErrorOf<P>>;
pub type ConstraintOf<P> = <P as ClosedProxy>::Constraint;
pub type DivergenceOf<P> = <ConstraintOf<P> as Constraint>::Divergence;
pub type ErrorOf<P> = <ConstraintOf<P> as Constraint>::Error;
pub type ExpressionOf<P> = Expression<P, ErrorOf<P>>;

/// A `Proxy` type that is closed over its primitive floating-point type and
/// constraint.
pub trait ClosedProxy: Sized {
    type Primitive: Float + Primitive;
    type Constraint: Constraint;
}

// TODO: By default, Serde serializes floating-point primitives representing
//       `NaN` and infinities as `"null"`. Moreover, Serde cannot deserialize
//       `"null"` as a floating-point primitive. This means that information is
//       lost when serializing and deserializing is impossible for non-real
//       values.
/// Serialization container.
///
/// This type is represented and serialized transparently as its inner type `T`.
/// `Proxy` uses this type for its own serialization and deserialization.
/// Importantly, this uses a conversion when deserializing that upholds the
/// constraints on proxy types, so it is not possible to deserialize a
/// floating-point value into a proxy type that does not support that value.
///
/// See the following for more context and details:
///
/// - https://github.com/serde-rs/serde/issues/642
/// - https://github.com/serde-rs/serde/issues/939
#[cfg(feature = "serialize-serde")]
#[derive(Deserialize, Serialize)]
#[serde(transparent)]
#[derive(Clone, Copy)]
#[repr(transparent)]
struct SerdeContainer<T> {
    inner: T,
}

#[cfg(feature = "serialize-serde")]
impl<T, C> From<Proxy<T, C>> for SerdeContainer<T>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn from(proxy: Proxy<T, C>) -> Self {
        SerdeContainer {
            inner: proxy.into_inner(),
        }
    }
}

/// Floating-point proxy that provides a total ordering, equivalence, hashing,
/// and constraints.
///
/// `Proxy` wraps primitive floating-point types and provides implementations
/// for numeric traits using a total ordering, including `Ord`, `Eq`, and
/// `Hash`. `Proxy` supports various constraints on the set of values that may
/// be represented and **panics if these constraints are violated in a numeric
/// operation.**
///
/// This type is re-exported but should not (and cannot) be used directly. Use
/// the type aliases `Total`, `NotNan`, and `Finite` instead.
///
/// # Total Ordering
///
/// All proxy types use the following total ordering:
///
/// $$-\infin<\cdots<0<\cdots<\infin<\text{NaN}$$
///
/// See the `cmp` module for a description of the total ordering used to
/// implement `Ord` and `Eq`.
///
/// # Constraints
///
/// Constraints restrict the set of values that a proxy may take by disallowing
/// certain classes or subsets of those values. If a constraint is violated
/// (because a proxy type would need to take a value it disallows), the
/// operation panics.
///
/// Constraints may disallow two broad classes of floating-point values:
/// infinities and `NaN`s. Constraints are exposed by the `Total`, `NotNan`, and
/// `Finite` type definitions. Note that `Total` uses a unit constraint, which
/// enforces no constraints at all and never panics.
#[cfg_attr(feature = "serialize-serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serialize-serde",
    serde(
        bound(
            deserialize = "T: serde::Deserialize<'de> + Float + Primitive, \
                           C: Constraint, \
                           C::Error: Display",
            serialize = "T: Float + Primitive + serde::Serialize, \
                         C: Constraint"
        ),
        try_from = "SerdeContainer<T>",
        into = "SerdeContainer<T>"
    )
)]
#[repr(transparent)]
pub struct Proxy<T, C> {
    inner: T,
    #[cfg_attr(feature = "serialize-serde", serde(skip))]
    phantom: PhantomData<fn() -> C>,
}

impl<T, C> Proxy<T, C> {
    pub(crate) const fn unchecked(inner: T) -> Self {
        Proxy {
            inner,
            phantom: PhantomData,
        }
    }

    /// Converts a proxy into a primitive floating-point value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use decorum::R64;
    ///
    /// fn f() -> R64 {
    /// #    use num_traits::Zero;
    /// #    R64::zero()
    ///     // ...
    /// }
    ///
    /// let x: f64 = f().into_inner();
    /// // The `From` and `Into` traits can also be used.
    /// let y: f64 = f().into();
    /// ```
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T, C> Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    // TODO: Update documentation for rename from `new` to `try_new`.
    /// Creates a proxy from a primitive floating-point value.
    ///
    /// This construction is also provided via `TryFrom`, but `new` must be used
    /// in generic code if the primitive floating-point type is unknown.
    ///
    /// # Errors
    ///
    /// Returns a `ConstraintViolation` error if the primitive floating-point
    /// value violates the constraints of the proxy. For `Total`, which has no
    /// constraints, the error type is `Infallible` and the construction cannot
    /// fail.
    ///
    /// # Examples
    ///
    /// Creating proxies from primitive floating-point values:
    ///
    /// ```rust
    /// use core::convert::TryInto;
    /// use decorum::R64;
    ///
    /// fn f(x: R64) -> R64 {
    ///     x * 2.0
    /// }
    ///
    /// let y = f(R64::new(2.0).unwrap());
    /// // The `TryFrom` and `TryInto` traits can also be used in some contexts.
    /// let z = f(2.0.try_into().unwrap());
    /// ```
    ///
    /// Creating a proxy with a failure:
    ///
    /// ```rust,should_panic
    /// use decorum::R64;
    ///
    /// // `R64` does not allow `NaN`s, but `0.0 / 0.0` produces a `NaN`.
    /// let x = R64::new(0.0 / 0.0).unwrap(); // Panics.
    /// ```
    pub fn try_new(inner: T) -> Result<Self, C::Error> {
        C::compliance(inner).map(|inner| Proxy {
            inner,
            phantom: PhantomData,
        })
    }

    /// Creates a proxy from a primitive floating-point value and asserts that
    /// constraints are not violated.
    ///
    /// For `Total`, which has no constraints, this function never fails.
    ///
    /// # Panics
    ///
    /// This construction panics if the primitive floating-point value violates
    /// the constraints of the proxy.
    ///
    /// # Examples
    ///
    /// Creating proxies from primitive floating-point values:
    ///
    /// ```rust
    /// use decorum::R64;
    ///
    /// fn f(x: R64) -> R64 {
    ///     x * 2.0
    /// }
    ///
    /// let y = f(R64::assert(2.0));
    /// ```
    ///
    /// Creating a proxy with a failure:
    ///
    /// ```rust,should_panic
    /// use decorum::R64;
    ///
    /// // `R64` does not allow `NaN`s, but `0.0 / 0.0` produces a `NaN`.
    /// let x = R64::assert(0.0 / 0.0); // Panics.
    /// ```
    pub fn assert(inner: T) -> Self {
        Self::try_new(inner).expect_constrained()
    }

    /// Converts a proxy into another proxy that is capable of representing a
    /// superset of the values that are members of its constraint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate num;
    /// use decorum::{N64, R64};
    /// use num::Zero;
    ///
    /// let x = R64::zero();
    /// let y = N64::from_subset(x);
    /// ```
    pub fn from_subset<C2>(other: Proxy<T, C2>) -> Self
    where
        C2: Constraint + SubsetOf<C>,
    {
        Self::unchecked(other.into_inner())
    }

    /// Converts a proxy into another proxy that is capable of representing a
    /// superset of the values that are members of its constraint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate num;
    /// use decorum::{N64, R64};
    /// use num::Zero;
    ///
    /// let x = R64::zero();
    /// let y: N64 = x.into_superset();
    /// ```
    pub fn into_superset<C2>(self) -> Proxy<T, C2>
    where
        C2: Constraint + SupersetOf<C>,
    {
        Proxy::unchecked(self.into_inner())
    }

    pub fn into_expression(self) -> ExpressionOf<Self> {
        Expression::from(self)
    }

    /// Converts a slice of primitive floating-point values into a slice of
    /// proxies.
    ///
    /// This conversion must check the constraints of the proxy against each
    /// floating-point value and so has `O(N)` time complexity.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the primitive floating-point values in the
    /// slice do not satisfy the constraints of the proxy.
    pub fn try_from_slice<'a>(slice: &'a [T]) -> Result<&'a [Self], C::Error> {
        slice
            .iter()
            .try_for_each(|inner| C::compliance(*inner).map(|_| ()))?;
        // SAFETY: `Proxy<T>` is `repr(transparent)` and has the same binary
        //         representation as its input type `T`. This means that it is
        //         safe to transmute `T` to `Proxy<T>`.
        Ok(unsafe { mem::transmute::<&'a [T], &'a [Self]>(slice) })
    }

    /// Converts a mutable slice of primitive floating-point values into a
    /// mutable slice of proxies.
    ///
    /// This conversion must check the constraints of the proxy against each
    /// floating-point value and so has `O(N)` time complexity.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the primitive floating-point values in the
    /// slice do not satisfy the constraints of the proxy.
    pub fn try_from_mut_slice<'a>(slice: &'a mut [T]) -> Result<&'a mut [Self], C::Error> {
        slice
            .iter()
            .try_for_each(|inner| C::compliance(*inner).map(|_| ()))?;
        // SAFETY: `Proxy<T>` is `repr(transparent)` and has the same binary
        //         representation as its input type `T`. This means that it is
        //         safe to transmute `T` to `Proxy<T>`.
        Ok(unsafe { mem::transmute::<&'a mut [T], &'a mut [Self]>(slice) })
    }
}

impl<T, C> Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    pub fn new(inner: T) -> BranchOf<Self> {
        C::branch(inner, |inner| Proxy {
            inner,
            phantom: PhantomData,
        })
    }

    pub(crate) fn map<F>(self, mut f: F) -> BranchOf<Self>
    where
        F: FnMut(T) -> T,
    {
        Self::new(f(self.into_inner()))
    }

    pub(crate) fn map_unchecked<F>(self, mut f: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        Proxy::unchecked(f(self.into_inner()))
    }

    pub(crate) fn zip_map<C2, F>(self, other: Proxy<T, C2>, mut f: F) -> BranchOf<Self>
    where
        C2: Constraint,
        F: FnMut(T, T) -> T,
    {
        Self::new(f(self.into_inner(), other.into_inner()))
    }

    pub(crate) fn zip_map_unchecked<C2, F>(self, other: Proxy<T, C2>, mut f: F) -> Self
    where
        C2: Constraint,
        F: FnMut(T, T) -> T,
    {
        Proxy::unchecked(f(self.into_inner(), other.into_inner()))
    }
}

impl<T> Total<T>
where
    T: Float + Primitive,
{
    /// Converts a slice of primitive floating-point values into a slice of
    /// `Total`s.
    ///
    /// Unlike `Proxy::try_from_slice`, this conversion is infallible and
    /// trivial and so has `O(1)` time complexity.
    pub fn from_slice<'a>(slice: &'a [T]) -> &'a [Self] {
        // SAFETY: `Proxy<T>` is `repr(transparent)` and has the same binary
        //         representation as its input type `T`. This means that it is
        //         safe to transmute `T` to `Proxy<T>`.
        unsafe { mem::transmute::<&'a [T], &'a [Self]>(slice) }
    }

    /// Converts a mutable slice of primitive floating-point values into a
    /// mutable slice of `Total`s.
    ///
    /// Unlike `Proxy::try_from_mut_slice`, this conversion is infallible and
    /// trivial and so has `O(1)` time complexity.
    pub fn from_mut_slice<'a>(slice: &'a mut [T]) -> &'a mut [Self] {
        // SAFETY: `Proxy<T>` is `repr(transparent)` and has the same binary
        //         representation as its input type `T`. This means that it is
        //         safe to transmute `T` to `Proxy<T>`.
        unsafe { mem::transmute::<&'a mut [T], &'a mut [Self]>(slice) }
    }
}

#[cfg(feature = "approx")]
impl<T, C> AbsDiffEq for Proxy<T, C>
where
    T: AbsDiffEq<Epsilon = T> + Float + Primitive,
    C: Constraint,
{
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::assert(T::default_epsilon())
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.into_inner()
            .abs_diff_eq(&other.into_inner(), epsilon.into_inner())
    }
}

impl<T, C> Add for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn add(self, other: Self) -> Self::Output {
        self.zip_map(other, Add::add)
    }
}

impl<T, C> Add<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn add(self, other: T) -> Self::Output {
        self.map(|inner| inner + other)
    }
}

// TODO:
//#[cfg(all(nightly, feature = "unstable"))]
//impl<T, C> Add<BranchOf<Self>> for Proxy<T, C>
//where
//    BranchOf<Self>: Try,
//    T: Float + Primitive,
//    C: Constraint,
//{
//    type Output = BranchOf<Self>;
//
//    fn add(self, other: BranchOf<Self>) -> Self::Output {
//        self.zip_map(other?, Add::add)
//    }
//}
//
//#[cfg(all(nightly, feature = "unstable"))]
//impl<T, C> Add<Proxy<T, C>> for BranchOf<Proxy<T, C>>
//where
//    Self: Try,
//    T: Float + Primitive,
//    C: Constraint,
//{
//    type Output = Self;
//
//    fn add(self, other: Proxy<T, C>) -> Self::Output {
//        self?.zip_map(other, Add::add)
//    }
//}

impl<T, C> AddAssign for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<T, C> AddAssign<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn add_assign(&mut self, other: T) {
        *self = self.map(|inner| inner + other);
    }
}

impl<T, C> AsRef<T> for Proxy<T, C> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T, C> BinaryReal for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Superset {
        self.zip_map(n, BinaryReal::div_euclid)
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Superset {
        self.zip_map(n, BinaryReal::rem_euclid)
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Self) -> Self::Superset {
        self.zip_map(n, BinaryReal::pow)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self::Superset {
        self.zip_map(base, BinaryReal::log)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Superset {
        self.zip_map(other, BinaryReal::hypot)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self::Superset {
        self.zip_map(other, BinaryReal::atan2)
    }
}

impl<T, C> BinaryReal<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Superset {
        self.map(|inner| BinaryReal::div_euclid(inner, n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Superset {
        self.map(|inner| BinaryReal::rem_euclid(inner, n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Superset {
        self.map(|inner| BinaryReal::pow(inner, n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Superset {
        self.map(|inner| BinaryReal::log(inner, base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Superset {
        self.map(|inner| BinaryReal::hypot(inner, other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Superset {
        self.map(|inner| BinaryReal::atan2(inner, other))
    }
}

impl<T, C> Bounded for Proxy<T, C>
where
    T: Float + Primitive,
{
    fn min_value() -> Self {
        Encoding::MIN_FINITE
    }

    fn max_value() -> Self {
        Encoding::MAX_FINITE
    }
}

impl<T, C> Clone for Proxy<T, C>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Proxy {
            inner: self.inner.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T, C> ClosedProxy for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Primitive = T;
    type Constraint = C;
}

impl<T, C> Codomain for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Superset = BranchOf<Self>;
}

impl<T, C> Copy for Proxy<T, C> where T: Copy {}

impl<T> Debug for Finite<T>
where
    T: Debug + Float + Primitive,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Finite").field(self.as_ref()).finish()
    }
}

impl<T> Debug for NotNan<T>
where
    T: Debug + Float + Primitive,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NotNan").field(self.as_ref()).finish()
    }
}

impl<T> Debug for Total<T>
where
    T: Debug + Float + Primitive,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Total").field(self.as_ref()).finish()
    }
}

impl<T, C> Default for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn default() -> Self {
        // There is no constraint that disallows real numbers such as zero.
        Self::unchecked(T::ZERO)
    }
}

impl<T, C> Display for Proxy<T, C>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, C> Div for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn div(self, other: Self) -> Self::Output {
        self.zip_map(other, Div::div)
    }
}

impl<T, C> Div<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn div(self, other: T) -> Self::Output {
        self.map(|inner| inner / other)
    }
}

impl<T, C> DivAssign for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn div_assign(&mut self, other: Self) {
        *self = *self / other
    }
}

impl<T, C> DivAssign<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn div_assign(&mut self, other: T) {
        *self = self.map(|inner| inner / other);
    }
}

impl<T, C> Encoding for Proxy<T, C>
where
    T: Float + Primitive,
{
    const MAX_FINITE: Self = Proxy::unchecked(T::MAX_FINITE);
    const MIN_FINITE: Self = Proxy::unchecked(T::MIN_FINITE);
    const MIN_POSITIVE_NORMAL: Self = Proxy::unchecked(T::MIN_POSITIVE_NORMAL);
    const EPSILON: Self = Proxy::unchecked(T::EPSILON);

    fn classify(self) -> FpCategory {
        T::classify(self.into_inner())
    }

    fn is_normal(self) -> bool {
        T::is_normal(self.into_inner())
    }

    fn is_sign_positive(self) -> bool {
        self.into_inner().is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.into_inner().is_sign_negative()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        T::integer_decode(self.into_inner())
    }
}

impl<T, C> Eq for Proxy<T, C> where T: Float + Primitive {}

// TODO: Bounds.
impl<T, C> FloatConst for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn E() -> Self {
        <Self as UnaryReal>::E
    }

    fn PI() -> Self {
        <Self as UnaryReal>::PI
    }

    fn SQRT_2() -> Self {
        <Self as UnaryReal>::SQRT_2
    }

    fn FRAC_1_PI() -> Self {
        <Self as UnaryReal>::FRAC_1_PI
    }

    fn FRAC_2_PI() -> Self {
        <Self as UnaryReal>::FRAC_2_PI
    }

    fn FRAC_1_SQRT_2() -> Self {
        <Self as UnaryReal>::FRAC_1_SQRT_2
    }

    fn FRAC_2_SQRT_PI() -> Self {
        <Self as UnaryReal>::FRAC_2_SQRT_PI
    }

    fn FRAC_PI_2() -> Self {
        <Self as UnaryReal>::FRAC_PI_2
    }

    fn FRAC_PI_3() -> Self {
        <Self as UnaryReal>::FRAC_PI_3
    }

    fn FRAC_PI_4() -> Self {
        <Self as UnaryReal>::FRAC_PI_4
    }

    fn FRAC_PI_6() -> Self {
        <Self as UnaryReal>::FRAC_PI_6
    }

    fn FRAC_PI_8() -> Self {
        <Self as UnaryReal>::FRAC_PI_8
    }

    fn LN_10() -> Self {
        <Self as UnaryReal>::LN_10
    }

    fn LN_2() -> Self {
        <Self as UnaryReal>::LN_2
    }

    fn LOG10_E() -> Self {
        <Self as UnaryReal>::LOG10_E
    }

    fn LOG2_E() -> Self {
        <Self as UnaryReal>::LOG2_E
    }
}

impl<T, C> ForeignFloat for Proxy<T, C>
where
    T: IntrinsicOrd + Float + ForeignFloat + Num + NumCast + Primitive,
    C: Constraint + Member<InfinitySet> + Member<NanSet>,
    C::Divergence: NonResidual<Self>,
{
    fn infinity() -> Self {
        Infinite::INFINITY
    }

    fn neg_infinity() -> Self {
        Infinite::NEG_INFINITY
    }

    fn is_infinite(self) -> bool {
        Infinite::is_infinite(self)
    }

    fn is_finite(self) -> bool {
        Infinite::is_finite(self)
    }

    fn nan() -> Self {
        Nan::NAN
    }

    fn is_nan(self) -> bool {
        Nan::is_nan(self)
    }

    fn max_value() -> Self {
        Encoding::MAX_FINITE
    }

    fn min_value() -> Self {
        Encoding::MIN_FINITE
    }

    fn min_positive_value() -> Self {
        Encoding::MIN_POSITIVE_NORMAL
    }

    fn epsilon() -> Self {
        Encoding::EPSILON
    }

    fn min(self, other: Self) -> Self {
        // Avoid panics by propagating `NaN`s for incomparable values.
        self.zip_map(other, cmp::min_or_undefined)
    }

    fn max(self, other: Self) -> Self {
        // Avoid panics by propagating `NaN`s for incomparable values.
        self.zip_map(other, cmp::max_or_undefined)
    }

    fn neg_zero() -> Self {
        -Self::ZERO
    }

    fn is_sign_positive(self) -> bool {
        Encoding::is_sign_positive(self.into_inner())
    }

    fn is_sign_negative(self) -> bool {
        Encoding::is_sign_negative(self.into_inner())
    }

    fn signum(self) -> Self {
        self.map(UnaryReal::signum)
    }

    fn abs(self) -> Self {
        self.map(UnaryReal::abs)
    }

    fn classify(self) -> FpCategory {
        Encoding::classify(self)
    }

    fn is_normal(self) -> bool {
        Encoding::is_normal(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        Encoding::integer_decode(self)
    }

    fn floor(self) -> Self {
        self.map(UnaryReal::floor)
    }

    fn ceil(self) -> Self {
        self.map(UnaryReal::ceil)
    }

    fn round(self) -> Self {
        self.map(UnaryReal::round)
    }

    fn trunc(self) -> Self {
        self.map(UnaryReal::trunc)
    }

    fn fract(self) -> Self {
        self.map(UnaryReal::fract)
    }

    fn recip(self) -> Self {
        self.map(UnaryReal::recip)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // TODO: This implementation requires a `ForeignFloat` bound and
        //       forwards to its `mul_add`. Consider supporting `mul_add` via a
        //       trait that is more specific to floating-point encoding than
        //       `BinaryReal` and friends.
        self.map(|inner| ForeignFloat::mul_add(inner, a.into_inner(), b.into_inner()))
    }

    #[cfg(feature = "std")]
    fn abs_sub(self, other: Self) -> Self {
        self.zip_map(other, |a, b| UnaryReal::abs(a - b))
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self {
        UnaryReal::powi(self, n)
    }

    #[cfg(feature = "std")]
    fn powf(self, n: Self) -> Self {
        BinaryReal::pow(self, n)
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self {
        UnaryReal::sqrt(self)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        UnaryReal::cbrt(self)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self {
        UnaryReal::exp(self)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self {
        UnaryReal::exp2(self)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self {
        UnaryReal::exp_m1(self)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self {
        BinaryReal::log(self, base)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self {
        UnaryReal::ln(self)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self {
        UnaryReal::log2(self)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self {
        UnaryReal::log10(self)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self {
        UnaryReal::ln_1p(self)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self {
        BinaryReal::hypot(self, other)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        UnaryReal::sin(self)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        UnaryReal::cos(self)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self {
        UnaryReal::tan(self)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self {
        UnaryReal::asin(self)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self {
        UnaryReal::acos(self)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        UnaryReal::atan(self)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self {
        BinaryReal::atan2(self, other)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        UnaryReal::sin_cos(self)
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        UnaryReal::sinh(self)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        UnaryReal::cosh(self)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        UnaryReal::tanh(self)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self {
        UnaryReal::asinh(self)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self {
        UnaryReal::acosh(self)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self {
        UnaryReal::atanh(self)
    }

    #[cfg(not(feature = "std"))]
    fn to_degrees(self) -> Self {
        UnaryReal::to_degrees(self)
    }

    #[cfg(not(feature = "std"))]
    fn to_radians(self) -> Self {
        UnaryReal::to_radians(self)
    }
}

impl<T> From<Finite<T>> for NotNan<T>
where
    T: Float + Primitive,
{
    fn from(other: Finite<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<'a, T> From<&'a T> for &'a Total<T>
where
    T: Float + Primitive,
{
    fn from(inner: &'a T) -> Self {
        // SAFETY: `Proxy<T>` is `repr(transparent)` and has the same binary
        //         representation as its input type `T`. This means that it is
        //         safe to transmute `T` to `Proxy<T>`.
        unsafe { &*(inner as *const T as *const Total<T>) }
    }
}

impl<'a, T> From<&'a mut T> for &'a mut Total<T>
where
    T: Float + Primitive,
{
    fn from(inner: &'a mut T) -> Self {
        // SAFETY: `Proxy<T>` is `repr(transparent)` and has the same binary
        //         representation as its input type `T`. This means that it is
        //         safe to transmute `T` to `Proxy<T>`.
        unsafe { &mut *(inner as *mut T as *mut Total<T>) }
    }
}

impl<T> From<Finite<T>> for Total<T>
where
    T: Float + Primitive,
{
    fn from(other: Finite<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<T> From<NotNan<T>> for Total<T>
where
    T: Float + Primitive,
{
    fn from(other: NotNan<T>) -> Self {
        Self::from_subset(other)
    }
}

impl<C> From<Proxy<f32, C>> for f32
where
    C: Constraint,
{
    fn from(proxy: Proxy<f32, C>) -> Self {
        proxy.into_inner()
    }
}

impl<C> From<Proxy<f64, C>> for f64
where
    C: Constraint,
{
    fn from(proxy: Proxy<f64, C>) -> Self {
        proxy.into_inner()
    }
}

impl<T> From<T> for Total<T>
where
    T: Float + Primitive,
{
    fn from(inner: T) -> Self {
        Self::unchecked(inner)
    }
}

impl<T, C> FromPrimitive for Proxy<T, C>
where
    T: Float + FromPrimitive + Primitive,
    C: Constraint,
{
    fn from_i8(value: i8) -> Option<Self> {
        T::from_i8(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_u8(value: u8) -> Option<Self> {
        T::from_u8(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_i16(value: i16) -> Option<Self> {
        T::from_i16(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_u16(value: u16) -> Option<Self> {
        T::from_u16(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_i32(value: i32) -> Option<Self> {
        T::from_i32(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_u32(value: u32) -> Option<Self> {
        T::from_u32(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_i64(value: i64) -> Option<Self> {
        T::from_i64(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_u64(value: u64) -> Option<Self> {
        T::from_u64(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_isize(value: isize) -> Option<Self> {
        T::from_isize(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_usize(value: usize) -> Option<Self> {
        T::from_usize(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_f32(value: f32) -> Option<Self> {
        T::from_f32(value).and_then(|inner| Proxy::try_new(inner).ok())
    }

    fn from_f64(value: f64) -> Option<Self> {
        T::from_f64(value).and_then(|inner| Proxy::try_new(inner).ok())
    }
}

impl<T, C> FromStr for Proxy<T, C>
where
    T: Float + FromStr + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    type Err = <T as FromStr>::Err;

    fn from_str(string: &str) -> Result<Self, Self::Err> {
        T::from_str(string).map(Self::new)
    }
}

impl<T, C> Hash for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        FloatHash::float_hash(self.as_ref(), state);
    }
}

impl<T, C> Infinite for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint + Member<InfinitySet>,
{
    const INFINITY: Self = Proxy::unchecked(T::INFINITY);
    const NEG_INFINITY: Self = Proxy::unchecked(T::NEG_INFINITY);

    fn is_infinite(self) -> bool {
        self.into_inner().is_infinite()
    }

    fn is_finite(self) -> bool {
        self.into_inner().is_finite()
    }
}

impl<T, C> LowerExp for Proxy<T, C>
where
    T: Float + LowerExp + Primitive,
    C: Constraint,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, C> Mul for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn mul(self, other: Self) -> Self::Output {
        self.zip_map(other, Mul::mul)
    }
}

impl<T, C> Mul<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn mul(self, other: T) -> Self::Output {
        self.map(|a| a * other)
    }
}

impl<T, C> MulAssign for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<T, C> MulAssign<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

impl<T, C> Nan for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint + Member<NanSet>,
{
    const NAN: Self = Proxy::unchecked(T::NAN);

    fn is_nan(self) -> bool {
        self.into_inner().is_nan()
    }
}

impl<T, C> Neg for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Proxy::unchecked(-self.into_inner())
    }
}

impl<T, C> Neg for ExpressionOf<Proxy<T, C>>
where
    T: Float + Primitive,
    C: Constraint<Divergence = TryExpression>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(|defined| -defined)
    }
}

impl<T, C> Num for Proxy<T, C>
where
    T: Float + Primitive + Num,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    // TODO: Differentiate between parse and contraint errors.
    type FromStrRadixErr = ();

    fn from_str_radix(source: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(source, radix)
            .map_err(|_| ())
            .and_then(|inner| Proxy::try_new(inner).map_err(|_| ()))
    }
}

impl<T, C> NumCast for Proxy<T, C>
where
    T: Float + NumCast + Primitive + ToPrimitive,
    C: Constraint,
{
    fn from<U>(value: U) -> Option<Self>
    where
        U: ToPrimitive,
    {
        T::from(value).and_then(|inner| Proxy::try_new(inner).ok())
    }
}

impl<T, C> One for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn one() -> Self {
        Proxy::unchecked(T::ONE)
    }
}

impl<T, C> Ord for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn cmp(&self, other: &Self) -> Ordering {
        FloatOrd::float_cmp(self.as_ref(), other.as_ref())
    }
}

impl<T, C> PartialEq for Proxy<T, C>
where
    T: Float + Primitive,
{
    fn eq(&self, other: &Self) -> bool {
        FloatEq::float_eq(self.as_ref(), other.as_ref())
    }
}

impl<T, C> PartialEq<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn eq(&self, other: &T) -> bool {
        if let Ok(other) = Self::try_new(*other) {
            Self::eq(self, &other)
        }
        else {
            false
        }
    }
}

impl<T, C> PartialOrd for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(FloatOrd::float_cmp(self.as_ref(), other.as_ref()))
    }
}

impl<T, C> PartialOrd<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Self::try_new(*other)
            .ok()
            .and_then(|other| Self::partial_cmp(self, &other))
    }
}

impl<T, C> Product for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn product<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(UnaryReal::ONE, |a, b| a * b)
    }
}

#[cfg(feature = "approx")]
impl<T, C> RelativeEq for Proxy<T, C>
where
    T: Float + Primitive + RelativeEq<Epsilon = T>,
    C: Constraint,
{
    fn default_max_relative() -> Self::Epsilon {
        Self::assert(T::default_max_relative())
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.into_inner().relative_eq(
            &other.into_inner(),
            epsilon.into_inner(),
            max_relative.into_inner(),
        )
    }
}

impl<T, C> Rem for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn rem(self, other: Self) -> Self::Output {
        self.zip_map(other, Rem::rem)
    }
}

impl<T, C> Rem<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn rem(self, other: T) -> Self::Output {
        self.map(|inner| inner % other)
    }
}

impl<T, C> RemAssign for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}

impl<T, C> RemAssign<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn rem_assign(&mut self, other: T) {
        *self = self.map(|inner| inner % other);
    }
}

impl<T, C> Signed for Proxy<T, C>
where
    T: Float + Primitive + Num,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn abs(&self) -> Self {
        self.map_unchecked(UnaryReal::abs)
    }

    #[cfg(feature = "std")]
    fn abs_sub(&self, other: &Self) -> Self {
        self.zip_map_unchecked(*other, |a, b| (a - b).abs())
    }

    #[cfg(not(feature = "std"))]
    fn abs_sub(&self, other: &Self) -> Self {
        self.zip_map_unchecked(*other, |a, b| {
            if a <= b {
                Zero::zero()
            }
            else {
                a - b
            }
        })
    }

    fn signum(&self) -> Self {
        self.map_unchecked(|inner| inner.signum())
    }

    fn is_positive(&self) -> bool {
        self.into_inner().is_positive()
    }

    fn is_negative(&self) -> bool {
        self.into_inner().is_negative()
    }
}

impl<T, C> Sub for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn sub(self, other: Self) -> Self::Output {
        self.zip_map(other, Sub::sub)
    }
}

impl<T, C> Sub<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Output = BranchOf<Self>;

    fn sub(self, other: T) -> Self::Output {
        self.map(|inner| inner - other)
    }
}

impl<T, C> SubAssign for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other
    }
}

impl<T, C> SubAssign<T> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn sub_assign(&mut self, other: T) {
        *self = self.map(|inner| inner - other)
    }
}

impl<T, C> Sum for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn sum<I>(input: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        input.fold(Zero::zero(), |a, b| a + b)
    }
}

impl<T, C> ToCanonicalBits for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Bits = <T as ToCanonicalBits>::Bits;

    fn to_canonical_bits(self) -> Self::Bits {
        self.inner.to_canonical_bits()
    }
}

impl<T, C> ToPrimitive for Proxy<T, C>
where
    T: Float + Primitive + ToPrimitive,
    C: Constraint,
{
    fn to_i8(&self) -> Option<i8> {
        self.into_inner().to_i8()
    }

    fn to_u8(&self) -> Option<u8> {
        self.into_inner().to_u8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.into_inner().to_i16()
    }

    fn to_u16(&self) -> Option<u16> {
        self.into_inner().to_u16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.into_inner().to_i32()
    }

    fn to_u32(&self) -> Option<u32> {
        self.into_inner().to_u32()
    }

    fn to_i64(&self) -> Option<i64> {
        self.into_inner().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.into_inner().to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.into_inner().to_isize()
    }

    fn to_usize(&self) -> Option<usize> {
        self.into_inner().to_usize()
    }

    fn to_f32(&self) -> Option<f32> {
        self.into_inner().to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.into_inner().to_f64()
    }
}

#[cfg(feature = "serialize-serde")]
impl<T, C> TryFrom<SerdeContainer<T>> for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    type Error = C::Error;

    fn try_from(container: SerdeContainer<T>) -> Result<Self, Self::Error> {
        Self::try_new(container.inner)
    }
}

#[cfg(feature = "approx")]
impl<T, C> UlpsEq for Proxy<T, C>
where
    T: Float + Primitive + UlpsEq<Epsilon = T>,
    C: Constraint,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.into_inner()
            .ulps_eq(&other.into_inner(), epsilon.into_inner(), max_ulps)
    }
}

impl<T, C> UnaryReal for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
{
    const ZERO: Self = Proxy::unchecked(UnaryReal::ZERO);
    const ONE: Self = Proxy::unchecked(UnaryReal::ONE);
    const E: Self = Proxy::unchecked(UnaryReal::E);
    const PI: Self = Proxy::unchecked(UnaryReal::PI);
    const FRAC_1_PI: Self = Proxy::unchecked(UnaryReal::FRAC_1_PI);
    const FRAC_2_PI: Self = Proxy::unchecked(UnaryReal::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = Proxy::unchecked(UnaryReal::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = Proxy::unchecked(UnaryReal::FRAC_PI_2);
    const FRAC_PI_3: Self = Proxy::unchecked(UnaryReal::FRAC_PI_3);
    const FRAC_PI_4: Self = Proxy::unchecked(UnaryReal::FRAC_PI_4);
    const FRAC_PI_6: Self = Proxy::unchecked(UnaryReal::FRAC_PI_6);
    const FRAC_PI_8: Self = Proxy::unchecked(UnaryReal::FRAC_PI_8);
    const SQRT_2: Self = Proxy::unchecked(UnaryReal::SQRT_2);
    const FRAC_1_SQRT_2: Self = Proxy::unchecked(UnaryReal::FRAC_1_SQRT_2);
    const LN_2: Self = Proxy::unchecked(UnaryReal::LN_2);
    const LN_10: Self = Proxy::unchecked(UnaryReal::LN_10);
    const LOG2_E: Self = Proxy::unchecked(UnaryReal::LOG2_E);
    const LOG10_E: Self = Proxy::unchecked(UnaryReal::LOG10_E);

    fn is_zero(self) -> bool {
        self.into_inner().is_zero()
    }

    fn is_one(self) -> bool {
        self.into_inner().is_zero()
    }

    fn is_positive(self) -> bool {
        self.into_inner().is_positive()
    }

    fn is_negative(self) -> bool {
        self.into_inner().is_negative()
    }

    #[cfg(feature = "std")]
    fn abs(self) -> Self {
        self.map_unchecked(UnaryReal::abs)
    }

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        self.map_unchecked(UnaryReal::signum)
    }

    fn floor(self) -> Self {
        self.map_unchecked(UnaryReal::floor)
    }

    fn ceil(self) -> Self {
        self.map_unchecked(UnaryReal::ceil)
    }

    fn round(self) -> Self {
        self.map_unchecked(UnaryReal::round)
    }

    fn trunc(self) -> Self {
        self.map_unchecked(UnaryReal::trunc)
    }

    fn fract(self) -> Self {
        self.map_unchecked(UnaryReal::fract)
    }

    fn recip(self) -> Self::Superset {
        self.map(UnaryReal::recip)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Superset {
        self.map(|inner| UnaryReal::powi(inner, n))
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Superset {
        self.map(UnaryReal::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map_unchecked(UnaryReal::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self::Superset {
        self.map(UnaryReal::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Superset {
        self.map(UnaryReal::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Superset {
        self.map(UnaryReal::exp_m1)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self::Superset {
        self.map(UnaryReal::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self::Superset {
        self.map(UnaryReal::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self::Superset {
        self.map(UnaryReal::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Superset {
        self.map(UnaryReal::ln_1p)
    }

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Superset {
        self.map(UnaryReal::to_degrees)
    }

    #[cfg(feature = "std")]
    fn to_radians(self) -> Self {
        self.map_unchecked(UnaryReal::to_radians)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map_unchecked(UnaryReal::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map_unchecked(UnaryReal::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self::Superset {
        self.map(UnaryReal::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self::Superset {
        self.map(UnaryReal::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self::Superset {
        self.map(UnaryReal::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map_unchecked(UnaryReal::atan)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.into_inner().sin_cos();
        (Proxy::unchecked(sin), Proxy::unchecked(cos))
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map_unchecked(UnaryReal::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map_unchecked(UnaryReal::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map_unchecked(UnaryReal::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Superset {
        self.map(UnaryReal::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Superset {
        self.map(UnaryReal::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Superset {
        self.map(UnaryReal::atanh)
    }
}

impl<T, C> UpperExp for Proxy<T, C>
where
    T: Float + Primitive + UpperExp,
    C: Constraint,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl<T, C> Zero for Proxy<T, C>
where
    T: Float + Primitive,
    C: Constraint,
    C::Divergence: NonResidual<Self>,
{
    fn zero() -> Self {
        Proxy::unchecked(T::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.into_inner().is_zero()
    }
}

macro_rules! impl_binary_operation {
    () => {
        with_binary_operations!(impl_binary_operation);
    };
    (operation => $trait:ident :: $method:ident) => {
        impl_binary_operation!(operation => $trait :: $method, |left, right| {
            right.map(|inner| $trait::$method(left, inner))
        });
    };
    (operation => $trait:ident :: $method:ident, |$left:ident, $right:ident| $f:block) => {
        macro_rules! impl_primitive_binary_operation {
            (primitive => $t:ty) => {
                impl<C> $trait<Proxy<$t, C>> for $t
                where
                    C: Constraint,
                {
                    type Output = BranchOf<Proxy<$t, C>>;

                    fn $method(self, other: Proxy<$t, C>) -> Self::Output {
                        let $left = self;
                        let $right = other;
                        $f
                    }
                }
            };
        }
        with_primitives!(impl_primitive_binary_operation);
    };
}
impl_binary_operation!();

/// Implements the `Real` trait from
/// [`num-traits`](https://crates.io/crates/num-traits) in terms of Decorum's
/// numeric traits. Does nothing if the `std` feature is disabled.
///
/// This is not generic, because the blanket implementation provided by
/// `num-traits` prevents a constraint-based implementation. Instead, this macro
/// must be applied manually to each proxy type exported by Decorum that is
/// `Real` but not `Float`.
///
/// See the following issues:
///
///   https://github.com/olson-sean-k/decorum/issues/10
///   https://github.com/rust-num/num-traits/issues/49
macro_rules! impl_foreign_real {
    () => {
        with_primitives!(impl_foreign_real);
    };
    (primitive => $t:ty) => {
        impl_foreign_real!(proxy => Finite, primitive => $t);
        impl_foreign_real!(proxy => NotNan, primitive => $t);
    };
    (proxy => $p:ident, primitive => $t:ty) => {
        #[cfg(feature = "std")]
        impl ForeignReal for $p<$t> {
            fn max_value() -> Self {
                Encoding::MAX_FINITE
            }

            fn min_value() -> Self {
                Encoding::MIN_FINITE
            }

            fn min_positive_value() -> Self {
                Encoding::MIN_POSITIVE_NORMAL
            }

            fn epsilon() -> Self {
                Encoding::EPSILON
            }

            fn min(self, other: Self) -> Self {
                // Avoid panics by propagating `NaN`s for incomparable values.
                self.zip_map(other, cmp::min_or_undefined)
            }

            fn max(self, other: Self) -> Self {
                // Avoid panics by propagating `NaN`s for incomparable values.
                self.zip_map(other, cmp::max_or_undefined)
            }

            fn is_sign_positive(self) -> bool {
                Encoding::is_sign_positive(self)
            }

            fn is_sign_negative(self) -> bool {
                Encoding::is_sign_negative(self)
            }

            fn signum(self) -> Self {
                Signed::signum(&self)
            }

            fn abs(self) -> Self {
                Signed::abs(&self)
            }

            fn floor(self) -> Self {
                UnaryReal::floor(self)
            }

            fn ceil(self) -> Self {
                UnaryReal::ceil(self)
            }

            fn round(self) -> Self {
                UnaryReal::round(self)
            }

            fn trunc(self) -> Self {
                UnaryReal::trunc(self)
            }

            fn fract(self) -> Self {
                UnaryReal::fract(self)
            }

            fn recip(self) -> Self {
                UnaryReal::recip(self)
            }

            fn mul_add(self, a: Self, b: Self) -> Self {
                self.map(|inner| inner.mul_add(a.into_inner(), b.into_inner()))
            }

            fn abs_sub(self, other: Self) -> Self {
                self.zip_map(other, ForeignFloat::abs_sub)
            }

            fn powi(self, n: i32) -> Self {
                UnaryReal::powi(self, n)
            }

            fn powf(self, n: Self) -> Self {
                BinaryReal::pow(self, n)
            }

            fn sqrt(self) -> Self {
                UnaryReal::sqrt(self)
            }

            fn cbrt(self) -> Self {
                UnaryReal::cbrt(self)
            }

            fn exp(self) -> Self {
                UnaryReal::exp(self)
            }

            fn exp2(self) -> Self {
                UnaryReal::exp2(self)
            }

            fn exp_m1(self) -> Self {
                UnaryReal::exp_m1(self)
            }

            fn log(self, base: Self) -> Self {
                BinaryReal::log(self, base)
            }

            fn ln(self) -> Self {
                UnaryReal::ln(self)
            }

            fn log2(self) -> Self {
                UnaryReal::log2(self)
            }

            fn log10(self) -> Self {
                UnaryReal::log10(self)
            }

            fn to_degrees(self) -> Self {
                self.map(ForeignFloat::to_degrees)
            }

            fn to_radians(self) -> Self {
                self.map(ForeignFloat::to_radians)
            }

            fn ln_1p(self) -> Self {
                UnaryReal::ln_1p(self)
            }

            fn hypot(self, other: Self) -> Self {
                BinaryReal::hypot(self, other)
            }

            fn sin(self) -> Self {
                UnaryReal::sin(self)
            }

            fn cos(self) -> Self {
                UnaryReal::cos(self)
            }

            fn tan(self) -> Self {
                UnaryReal::tan(self)
            }

            fn asin(self) -> Self {
                UnaryReal::asin(self)
            }

            fn acos(self) -> Self {
                UnaryReal::acos(self)
            }

            fn atan(self) -> Self {
                UnaryReal::atan(self)
            }

            fn atan2(self, other: Self) -> Self {
                BinaryReal::atan2(self, other)
            }

            fn sin_cos(self) -> (Self, Self) {
                UnaryReal::sin_cos(self)
            }

            fn sinh(self) -> Self {
                UnaryReal::sinh(self)
            }

            fn cosh(self) -> Self {
                UnaryReal::cosh(self)
            }

            fn tanh(self) -> Self {
                UnaryReal::tanh(self)
            }

            fn asinh(self) -> Self {
                UnaryReal::asinh(self)
            }

            fn acosh(self) -> Self {
                UnaryReal::acosh(self)
            }

            fn atanh(self) -> Self {
                UnaryReal::atanh(self)
            }
        }
    };
}
impl_foreign_real!();

// `TryFrom` cannot be implemented over an open type `T` and cannot be
// implemented for general constraints, because it would conflict with the
// `From` implementation for `Total`.
macro_rules! impl_try_from {
    () => {
        with_primitives!(impl_try_from);
    };
    (primitive => $t:ty) => {
        impl_try_from!(proxy => Finite, primitive => $t);
        impl_try_from!(proxy => NotNan, primitive => $t);
    };
    (proxy => $p:ident, primitive => $t:ty) => {
        impl<D> TryFrom<$t> for $p<$t, D>
        where
            D: Divergence,
        {
            type Error = ConstraintViolation;

            fn try_from(inner: $t) -> Result<Self, Self::Error> {
                Self::try_new(inner)
            }
        }

        impl<'a, D> TryFrom<&'a $t> for &'a $p<$t, D>
        where
            D: Divergence,
        {
            type Error = ConstraintViolation;

            fn try_from(inner: &'a $t) -> Result<Self, Self::Error> {
                ConstraintOf::<$p<$t, D>>::compliance(*inner).map(|_| {
                    // SAFETY: `Proxy<T>` is `repr(transparent)` and has the
                    //         same binary representation as its input type `T`.
                    //         This means that it is safe to transmute `T` to
                    //         `Proxy<T>`.
                    unsafe { mem::transmute::<&'a $t, Self>(inner) }
                })
            }
        }

        impl<'a, D> TryFrom<&'a mut $t> for &'a mut $p<$t, D>
        where
            D: Divergence,
        {
            type Error = ConstraintViolation;

            fn try_from(inner: &'a mut $t) -> Result<Self, Self::Error> {
                ConstraintOf::<$p<$t, D>>::compliance(*inner).map(move |_| {
                    // SAFETY: `Proxy<T>` is `repr(transparent)` and has the
                    //         same binary representation as its input type `T`.
                    //         This means that it is safe to transmute `T` to
                    //         `Proxy<T>`.
                    unsafe { mem::transmute::<&'a mut $t, Self>(inner) }
                })
            }
        }
    };
}
impl_try_from!();

#[cfg(test)]
mod tests {
    use core::convert::TryInto;

    use crate::{Finite, Float, Infinite, Nan, NotNan, Real, Total, UnaryReal, N32, R32};

    #[test]
    fn total_no_panic_on_inf() {
        let x: Total<f32> = 1.0.into();
        let y = x / 0.0;
        assert!(Infinite::is_infinite(y));
    }

    #[test]
    fn total_no_panic_on_nan() {
        let x: Total<f32> = 0.0.into();
        let y = x / 0.0;
        assert!(Nan::is_nan(y));
    }

    // This is the most comprehensive and general test of reference conversions,
    // as there are no failure conditions. Other similar tests focus solely on
    // success or failure, not completeness of the APIs under test. This test is
    // an ideal Miri target.
    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn total_no_panic_from_ref_slice() {
        let x = 0.0f64 / 0.0;
        let y: &Total<_> = (&x).into();
        assert!(y.is_nan());

        let mut x = 0.0f64;
        let y: &mut Total<_> = (&mut x).into();
        *y = (0.0f64 / 0.0).into();
        assert!(y.is_nan());

        let xs = [0.0f64, 1.0];
        let ys = Total::from_slice(&xs);
        assert_eq!(ys, &[0.0f64, 1.0]);

        let xs = [0.0f64, 1.0];
        let ys = Total::from_slice(&xs);
        assert_eq!(ys, &[0.0f64, 1.0]);
    }

    #[test]
    fn notnan_no_panic_on_inf() {
        let x: N32 = 1.0.try_into().unwrap();
        let y = x / 0.0;
        assert!(Infinite::is_infinite(y));
    }

    #[test]
    #[should_panic]
    fn notnan_panic_on_nan() {
        let x: N32 = 0.0.try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    fn notnan_no_panic_from_inf_ref_slice() {
        let x = 1.0f64 / 0.0;
        let y: &NotNan<_> = (&x).try_into().unwrap();
        assert!(y.is_infinite());

        let xs = [0.0f64, 1.0 / 0.0];
        let ys = NotNan::try_from_slice(&xs).unwrap();
        assert_eq!(ys, &[0.0f64, Infinite::INFINITY]);
    }

    #[test]
    #[should_panic]
    #[allow(clippy::zero_divided_by_zero)]
    fn notnan_panic_from_nan_ref() {
        let x = 0.0f64 / 0.0;
        let _: &NotNan<_> = (&x).try_into().unwrap();
    }

    #[test]
    #[should_panic]
    #[allow(clippy::zero_divided_by_zero)]
    fn notnan_panic_from_nan_slice() {
        let xs = [1.0f64, 0.0f64 / 0.0];
        let _ = NotNan::<f64>::try_from_slice(&xs).unwrap();
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_nan() {
        let x: R32 = 0.0.try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_inf() {
        let x: R32 = 1.0.try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_on_neg_inf() {
        let x: R32 = (-1.0).try_into().unwrap();
        let _ = x / 0.0;
    }

    #[test]
    #[should_panic]
    fn finite_panic_from_inf_ref() {
        let x = 1.0f64 / 0.0;
        let _: &Finite<_> = (&x).try_into().unwrap();
    }

    #[test]
    #[should_panic]
    fn finite_panic_from_inf_slice() {
        let xs = [1.0f64, 1.0f64 / 0.0];
        let _ = Finite::<f64>::try_from_slice(&xs).unwrap();
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn total_nan_eq() {
        let x: Total<f32> = (0.0 / 0.0).into();
        let y: Total<f32> = (0.0 / 0.0).into();
        assert_eq!(x, y);

        let z: Total<f32> = (<f32 as Infinite>::INFINITY + <f32 as Infinite>::NEG_INFINITY).into();
        assert_eq!(x, z);

        #[cfg(feature = "std")]
        {
            let w: Total<f32> = (UnaryReal::sqrt(-1.0f32)).into();
            assert_eq!(x, w);
        }
    }

    #[test]
    #[allow(clippy::cmp_nan)]
    #[allow(clippy::eq_op)]
    #[allow(clippy::float_cmp)]
    #[allow(clippy::zero_divided_by_zero)]
    fn cmp_proxy_primitive() {
        // Compare a canonicalized `NaN` with a primitive `NaN` with a
        // different representation.
        let x: Total<f32> = (0.0 / 0.0).into();
        assert_eq!(x, f32::sqrt(-1.0));

        // Compare a canonicalized `INF` with a primitive `NaN`.
        let y: Total<f32> = (1.0 / 0.0).into();
        assert!(y < (0.0 / 0.0));

        // Compare a proxy that disallows `INF` to a primitive `INF`.
        let z: R32 = 0.0.try_into().unwrap();
        assert_eq!(z.partial_cmp(&(1.0 / 0.0)), None);
    }

    #[test]
    fn sum() {
        let xs = [
            1.0.try_into().unwrap(),
            2.0.try_into().unwrap(),
            3.0.try_into().unwrap(),
        ];
        assert_eq!(xs.iter().cloned().sum::<R32>(), R32::assert(6.0));
    }

    #[test]
    fn product() {
        let xs = [
            1.0.try_into().unwrap(),
            2.0.try_into().unwrap(),
            3.0.try_into().unwrap(),
        ];
        assert_eq!(xs.iter().cloned().product::<R32>(), R32::assert(6.0),);
    }

    // TODO: This test is questionable.
    #[test]
    fn impl_traits() {
        fn as_float<T>(_: T)
        where
            T: Float,
        {
        }

        fn as_infinite<T>(_: T)
        where
            T: Infinite,
        {
        }

        fn as_nan<T>(_: T)
        where
            T: Nan,
        {
        }

        fn as_real<T>(_: T)
        where
            T: Real,
        {
        }

        let finite = Finite::<f32>::default();
        as_real(finite);

        let notnan = NotNan::<f32>::default();
        as_infinite(notnan);
        as_real(notnan);

        let ordered = Total::<f32>::default();
        as_float(ordered);
        as_infinite(ordered);
        as_nan(ordered);
        as_real(ordered);
    }

    #[test]
    fn fmt() {
        let x: Total<f32> = 1.0.into();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", x);
        let y: NotNan<f32> = 1.0.try_into().unwrap();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", y);
        let z: Finite<f32> = 1.0.try_into().unwrap();
        format_args!("{0} {0:e} {0:E} {0:?} {0:#?}", z);
    }

    #[cfg(feature = "serialize-serde")]
    #[test]
    fn deserialize() {
        assert_eq!(
            R32::assert(1.0),
            serde_json::from_str::<R32>("1.0").unwrap()
        );
    }

    #[cfg(feature = "serialize-serde")]
    #[test]
    #[should_panic]
    fn deserialize_panic_on_violation() {
        // TODO: See `SerdeContainer`. This does not test a value that violates
        //       `N32`'s constraints; instead, this simply fails to deserialize
        //       `f32` from `"null"`.
        let _: N32 = serde_json::from_str("null").unwrap();
    }

    #[cfg(feature = "serialize-serde")]
    #[test]
    fn serialize() {
        assert_eq!("1.0", serde_json::to_string(&N32::assert(1.0)).unwrap());
        // TODO: See `SerdeContainer`.
        assert_eq!("null", serde_json::to_string(&N32::INFINITY).unwrap());
    }
}
