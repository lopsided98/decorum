//! Constraints on the members of floating-point values that proxy types may
//! represent.

use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(not(feature = "std"))]
use core::fmt::{self, Display, Formatter};
use core::marker::PhantomData;
#[cfg(feature = "std")]
use thiserror::Error;

use crate::error::ErrorMode;
use crate::{Float, Primitive};

const VIOLATION_MESSAGE: &str = "floating-point constraint violated";

#[cfg_attr(feature = "std", derive(Error))]
#[cfg_attr(feature = "std", error("{}", VIOLATION_MESSAGE))]
#[derive(Clone, Copy, Debug)]
pub struct ConstraintViolation;

// When the `std` feature is enabled, the `thiserror` crate is used to implement
// `Display`.
#[cfg(not(feature = "std"))]
impl Display for ConstraintViolation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", VIOLATION_MESSAGE)
    }
}

pub trait ExpectConstrained<T>: Sized {
    fn expect_constrained(self) -> T;
}

impl<T, E> ExpectConstrained<T> for Result<T, E>
where
    E: Debug,
{
    #[cfg(not(feature = "std"))]
    fn expect_constrained(self) -> T {
        self.expect(VIOLATION_MESSAGE)
    }

    #[cfg(feature = "std")]
    fn expect_constrained(self) -> T {
        // When the `std` feature is enabled, `ConstraintViolation` implements
        // `Error` and an appropriate error message is displayed when
        // unwrapping.
        self.unwrap()
    }
}

pub enum RealSet {}
pub enum InfinitySet {}
pub enum NanSet {}

pub trait Member<T> {}

pub trait SupersetOf<P> {}

pub trait SubsetOf<P> {}

impl<P, Q> SubsetOf<Q> for P where Q: SupersetOf<P> {}

/// Describes constraints on the set of floating-point values that a proxy type
/// may represent.
///
/// This trait expresses a constraint by defining an error and emitting that
/// error from its `check` function if a primitive floating-point value violates
/// the constraint. Note that constraints require `Member<RealSet>`, meaning
/// that the set of real numbers must always be supported and is implied.
pub trait Constraint: Member<RealSet> {
    type ErrorMode: ErrorMode;
    type Error: Debug;

    /// Determines if a primitive floating-point value satisfies the constraint.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` if the primitive floating-point value violates the
    /// constraint.
    fn noncompliance<T>(inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive;

    fn compliance<T>(inner: T) -> Result<T, Self::Error>
    where
        T: Float + Primitive,
    {
        Self::noncompliance(inner).map_or(Ok(inner), |error| Err(error))
    }

    fn branch<T, U, F>(inner: T, f: F) -> <Self::ErrorMode as ErrorMode>::Branch<U, Self::Error>
    where
        T: Float + Primitive,
        F: FnOnce(T) -> U,
    {
        match Self::noncompliance(inner) {
            Some(error) => Self::ErrorMode::from_residual(error),
            _ => Self::ErrorMode::from_output(f(inner)),
        }
    }
}

#[derive(Debug)]
pub enum UnitConstraint {}

impl Constraint for UnitConstraint {
    type ErrorMode = Infallible;
    type Error = Infallible;

    fn noncompliance<T>(_inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive,
    {
        None
    }
}

impl Member<InfinitySet> for UnitConstraint {}

impl Member<NanSet> for UnitConstraint {}

impl Member<RealSet> for UnitConstraint {}

impl<M> SupersetOf<FiniteConstraint<M>> for UnitConstraint {}

impl<M> SupersetOf<NotNanConstraint<M>> for UnitConstraint {}

/// Disallows `NaN`s.
#[derive(Debug)]
pub struct NotNanConstraint<M> {
    phantom: PhantomData<fn() -> M>,
}

impl<M> Constraint for NotNanConstraint<M>
where
    M: ErrorMode,
{
    type ErrorMode = M;
    type Error = ConstraintViolation;

    fn noncompliance<T>(inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive,
    {
        inner.is_nan().then(|| ConstraintViolation)
    }
}

impl<M> Member<InfinitySet> for NotNanConstraint<M> {}

impl<M> Member<RealSet> for NotNanConstraint<M> {}

impl<M> SupersetOf<FiniteConstraint<M>> for NotNanConstraint<M> {}

/// Disallows `NaN`s and infinities.
#[derive(Debug)]
pub struct FiniteConstraint<M> {
    phantom: PhantomData<fn() -> M>,
}

impl<M> Constraint for FiniteConstraint<M>
where
    M: ErrorMode,
{
    type ErrorMode = M;
    type Error = ConstraintViolation;

    fn noncompliance<T>(inner: T) -> Option<Self::Error>
    where
        T: Float + Primitive,
    {
        (inner.is_nan() || inner.is_infinite()).then(|| ConstraintViolation)
    }
}

impl<M> Member<RealSet> for FiniteConstraint<M> {}
