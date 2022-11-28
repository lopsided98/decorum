//! Constraints on the members of floating-point values that proxy types may
//! represent.

use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(not(feature = "std"))]
use core::fmt::{self, Display, Formatter};
#[cfg(feature = "std")]
use thiserror::Error;

use crate::proxy::ClosedProxy;
use crate::{Float, Primitive};

const VIOLATION_MESSAGE: &str = "floating-point constraint violated";

pub type Error<T> = <<T as ClosedProxy>::Constraint as Constraint>::Error;

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
    type Error: Debug;

    /// Determines if a primitive floating-point value satisfies the constraint.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` if the primitive floating-point value violates the
    /// constraint.
    fn check<T>(inner: &T) -> Result<(), Self::Error>
    where
        T: Float + Primitive;
}

#[derive(Debug)]
pub enum UnitConstraint {}

impl Constraint for UnitConstraint {
    type Error = Infallible;

    fn check<T>(_: &T) -> Result<(), Self::Error>
    where
        T: Float + Primitive,
    {
        Ok(())
    }
}

impl Member<InfinitySet> for UnitConstraint {}

impl Member<NanSet> for UnitConstraint {}

impl Member<RealSet> for UnitConstraint {}

impl SupersetOf<FiniteConstraint> for UnitConstraint {}

impl SupersetOf<NotNanConstraint> for UnitConstraint {}

/// Disallows `NaN`s.
#[derive(Debug)]
pub enum NotNanConstraint {}

impl Constraint for NotNanConstraint {
    type Error = ConstraintViolation;

    fn check<T>(inner: &T) -> Result<(), Self::Error>
    where
        T: Float + Primitive,
    {
        if inner.is_nan() {
            Err(ConstraintViolation)
        }
        else {
            Ok(())
        }
    }
}

impl Member<InfinitySet> for NotNanConstraint {}

impl Member<RealSet> for NotNanConstraint {}

impl SupersetOf<FiniteConstraint> for NotNanConstraint {}

/// Disallows `NaN`s and infinities.
#[derive(Debug)]
pub enum FiniteConstraint {}

impl Constraint for FiniteConstraint {
    type Error = ConstraintViolation;

    fn check<T>(inner: &T) -> Result<(), Self::Error>
    where
        T: Float + Primitive,
    {
        if inner.is_nan() || inner.is_infinite() {
            Err(ConstraintViolation)
        }
        else {
            Ok(())
        }
    }
}

impl Member<RealSet> for FiniteConstraint {}
