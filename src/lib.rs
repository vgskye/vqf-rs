#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "f32"), allow(clippy::unnecessary_cast))]
//! An implementation of the VQF IMU orientation estimation filter in pure Rust.
//!
//! This is, currently, a pretty direct port of the C++ implemenataion in <https://github.com/dlaidig/vqf>;
//! further efforts to make the code more idiomatic may be helpful.
//!
//! The main entry point for this crate is [`VQF`]; look there to get started.
//!
//! This crate optionally supports `no_std`; the `libm` crate feature is required in `no_std` environments.

#[cfg(feature = "f32")]
/// Typedef for the floating-point data type used for most operations.
///
/// By default, all floating-point calculations are performed using `f64`. Enable the `f32` crate feature to
/// change this type to `f32`. Note that the Butterworth filter implementation will always use `f64` as
/// using `f32` can cause numeric issues.
pub type Float = f32;
#[cfg(not(feature = "f32"))]
/// Typedef for the floating-point data type used for most operations.
///
/// By default, all floating-point calculations are performed using `f64`. Enable the `f32` crate feature to
/// change this type to `f32`. Note that the Butterworth filter implementation will always use `f64` as
/// using `f32` can cause numeric issues.
pub type Float = f64;

#[cfg(feature = "std")]
type Math<T> = T;
#[cfg(feature = "libm")]
type Math<T> = libm::Libm<T>;

#[cfg(feature = "f32")]
use core::f32::consts as fc;
#[cfg(not(feature = "f32"))]
use core::f64::consts as fc;
use core::{
    f64::consts as f64c,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

fn flatten<const N: usize, const M: usize, const R: usize>(value: &mut [[Float; N]; M]) -> &mut [Float; R] {
    assert_eq!(N * M, R);
    // SAFETY: `[[T; N]; M]` is layout-identical to `[T; N * M]`
    unsafe { core::mem::transmute(value) }
}

/// A quaternion.
///
/// As this type was made solely for internal purposes, external usage seems ill-advised.
#[derive(Clone, Copy, Default)]
pub struct Quaternion(pub Float, pub Float, pub Float, pub Float);

impl Mul for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Self) -> Self::Output {
        let w = self.0 * rhs.0 - self.1 * rhs.1 - self.2 * rhs.2 - self.3 * rhs.3;
        let x = self.0 * rhs.1 + self.1 * rhs.0 + self.2 * rhs.3 - self.3 * rhs.2;
        let y = self.0 * rhs.2 - self.1 * rhs.3 + self.2 * rhs.0 + self.3 * rhs.1;
        let z = self.0 * rhs.3 + self.1 * rhs.2 - self.2 * rhs.1 + self.3 * rhs.0;
        Self(w, x, y, z)
    }
}

impl MulAssign for Quaternion {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl From<[Float; 4]> for Quaternion {
    fn from(value: [Float; 4]) -> Self {
        Self(value[0], value[1], value[2], value[3])
    }
}

impl Quaternion {
    pub fn norm(&self) -> Float {
        Math::<Float>::sqrt(square(self.0) + square(self.1) + square(self.2) + square(self.3))
    }

    pub fn normalize(&mut self) {
        let n = self.norm();
        if n < Float::EPSILON {
            return;
        }
        self.0 /= n;
        self.1 /= n;
        self.2 /= n;
        self.3 /= n;
    }

    pub fn rotate(&self, v: [Float; 3]) -> [Float; 3] {
        let x = (1.0 - 2.0 * self.2 * self.2 - 2.0 * self.3 * self.3) * v[0]
            + 2.0 * v[1] * (self.2 * self.1 - self.0 * self.3)
            + 2.0 * v[2] * (self.0 * self.2 + self.3 * self.1);
        let y = 2.0 * v[0] * (self.0 * self.3 + self.2 * self.1)
            + v[1] * (1.0 - 2.0 * self.1 * self.1 - 2.0 * self.3 * self.3)
            + 2.0 * v[2] * (self.2 * self.3 - self.1 * self.0);
        let z = 2.0 * v[0] * (self.3 * self.1 - self.0 * self.2)
            + 2.0 * v[1] * (self.0 * self.1 + self.3 * self.2)
            + v[2] * (1.0 - 2.0 * self.1 * self.1 - 2.0 * self.2 * self.2);
        [x, y, z]
    }

    pub fn apply_delta(&self, delta: Float) -> Quaternion {
        let c = Math::<Float>::cos(delta / 2.0);
        let s = Math::<Float>::sin(delta / 2.0);
        let w = c * self.0 - s * self.3;
        let x = c * self.1 - s * self.2;
        let y = c * self.2 + s * self.1;
        let z = c * self.3 + s * self.0;
        Self(w, x, y, z)
    }
}

/// A fixed-size matrix.
///
/// As this type was made solely for internal purposes, external usage seems ill-advised.
#[derive(Clone, Copy)]
pub struct Matrix<const W: usize, const H: usize>(pub [[Float; W]; H]);

impl<const W: usize, const H: usize> Default for Matrix<W, H> {
    fn default() -> Self {
        Self([[0.0; W]; H])
    }
}

impl<const W: usize, const H: usize> From<[[Float; W]; H]> for Matrix<W, H> {
    fn from(value: [[Float; W]; H]) -> Self {
        Self(value)
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Matrix<P, N>> for Matrix<N, M> {
    type Output = Matrix<P, M>;

    fn mul(self, rhs: Matrix<P, N>) -> Self::Output {
        let mut out: Matrix<P, M> = Default::default();
        for i in 0..M {
            for j in 0..P {
                let mut val = 0.0;
                for k in 0..N {
                    val += self.0[i][k] * rhs.0[k][j];
                }
                out.0[i][j] = val;
            }
        }
        out
    }
}

impl<const X: usize> MulAssign for Matrix<X, X> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const W: usize, const H: usize> Add for Matrix<W, H> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out: Self = Default::default();
        for i in 0..W {
            for j in 0..H {
                out.0[j][i] = self.0[j][i] + rhs.0[j][i];
            }
        }
        out
    }
}

impl<const W: usize, const H: usize> AddAssign for Matrix<W, H> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const W: usize, const H: usize> Sub for Matrix<W, H> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out: Self = Default::default();
        for i in 0..W {
            for j in 0..H {
                out.0[j][i] = self.0[j][i] - rhs.0[j][i];
            }
        }
        out
    }
}

impl<const W: usize, const H: usize> SubAssign for Matrix<W, H> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const W: usize, const H: usize> Matrix<W, H> {
    pub fn transpose(self) -> Matrix<H, W> {
        let mut out: Matrix<H, W> = Default::default();
        for i in 0..W {
            for j in 0..H {
                out.0[i][j] = self.0[j][i];
            }
        }
        out
    }
}

impl Matrix<3, 3> {
    pub fn invert(self) -> Option<Self> {
        // in = [a b c; d e f; g h i]
        let a =
            self.0[1][1] as f64 * self.0[2][2] as f64 - self.0[1][2] as f64 * self.0[2][1] as f64; // (e*i - f*h)
        let d =
            self.0[0][2] as f64 * self.0[2][1] as f64 - self.0[0][1] as f64 * self.0[2][2] as f64; // -(b*i - c*h)
        let g =
            self.0[0][1] as f64 * self.0[1][2] as f64 - self.0[0][2] as f64 * self.0[1][1] as f64; // (b*f - c*e)
        let b =
            self.0[1][2] as f64 * self.0[2][0] as f64 - self.0[1][0] as f64 * self.0[2][2] as f64; // -(d*i - f*g)
        let e =
            self.0[0][0] as f64 * self.0[2][2] as f64 - self.0[0][2] as f64 * self.0[2][0] as f64; // (a*i - c*g)
        let h =
            self.0[0][2] as f64 * self.0[1][0] as f64 - self.0[0][0] as f64 * self.0[1][2] as f64; // -(a*f - c*d)
        let c =
            self.0[1][0] as f64 * self.0[2][1] as f64 - self.0[1][1] as f64 * self.0[2][0] as f64; // (d*h - e*g)
        let f =
            self.0[0][1] as f64 * self.0[2][0] as f64 - self.0[0][0] as f64 * self.0[2][1] as f64; // -(a*h - b*g)
        let i =
            self.0[0][0] as f64 * self.0[1][1] as f64 - self.0[0][1] as f64 * self.0[1][0] as f64; // (a*e - b*d)

        let det = self.0[0][0] as f64 * a + self.0[0][1] as f64 * b + self.0[0][2] as f64 * c; // a*A + b*B + c*C;

        if (-f64::EPSILON..=f64::EPSILON).contains(&det) {
            return None;
        }

        // out = [A D G; B E H; C F I]/det
        Some(
            [
                [(a / det) as Float, (d / det) as Float, (g / det) as Float],
                [(b / det) as Float, (e / det) as Float, (h / det) as Float],
                [(c / det) as Float, (f / det) as Float, (i / det) as Float],
            ]
            .into(),
        )
    }
}

/// Struct containing all tuning parameters used by the VQF class.
///
/// The parameters influence the behavior of the algorithm and are independent of the sampling rate of the IMU data. The
/// constructor sets all parameters to the default values.
///
/// The parameters [`motion_bias_est_enabled`](Self::motion_bias_est_enabled), [`rest_bias_est_enabled`](Self::rest_bias_est_enabled), and [mag_dist_rejection_enabled`](Self::mag_dist_rejection_enabled) can be used to enable/disable
/// the main features of the VQF algorithm. The time constants [`tau_acc`](Self::tau_acc) and [`tau_mag`](Self::tau_mag) can be tuned to change the trust on
/// the accelerometer and magnetometer measurements, respectively. The remaining parameters influence bias estimation
/// and magnetometer rejection.
#[cfg_attr(doc, doc = include_str!("../katex.html"))]
#[derive(Clone, Copy)]
pub struct Params {
    /// Time constant $\tau_\mathrm{acc}$ for accelerometer low-pass filtering in seconds.
    ///
    /// Small values for $\tau_\mathrm{acc}$ imply trust on the accelerometer measurements and while large values of
    /// $\tau_\mathrm{acc}$ imply trust on the gyroscope measurements.
    ///
    /// The time constant $\tau_\mathrm{acc}$ corresponds to the cutoff frequency $f_\mathrm{c}$ of the
    /// second-order Butterworth low-pass filter as follows: $f_\mathrm{c} = \frac{\sqrt{2}}{2\pi\tau_\mathrm{acc}}$.
    ///
    /// Default value: 3.0 s
    pub tau_acc: Float,

    /// Time constant $\tau_\mathrm{mag}$ for magnetometer update in seconds.
    ///
    /// Small values for $\tau_\mathrm{mag}$ imply trust on the magnetometer measurements and while large values of
    /// $\tau_\mathrm{mag}$ imply trust on the gyroscope measurements.
    ///
    /// The time constant $\tau_\mathrm{mag}$ corresponds to the cutoff frequency $f_\mathrm{c}$ of the
    /// first-order low-pass filter for the heading correction as follows:
    /// $f_\mathrm{c} = \frac{1}{2\pi\tau_\mathrm{mag}}$.
    ///
    /// Default value: 9.0 s
    pub tau_mag: Float,

    #[cfg(feature = "motion-bias-estimation")]
    /// Enables gyroscope bias estimation during motion phases.
    ///
    /// If set to true (default), gyroscope bias is estimated based on the inclination correction only, i.e. without
    /// using magnetometer measurements.
    pub motion_bias_est_enabled: bool,

    /// Enables rest detection and gyroscope bias estimation during rest phases.
    ///
    /// If set to true (default), phases in which the IMU is at rest are detected. During rest, the gyroscope bias
    /// is estimated from the low-pass filtered gyroscope readings.
    pub rest_bias_est_enabled: bool,

    /// Enables magnetic disturbance detection and magnetic disturbance rejection.
    ///
    /// If set to true (default), the magnetic field is analyzed. For short disturbed phases, the magnetometer-based
    /// correction is disabled totally. If the magnetic field is always regarded as disturbed or if the duration of
    /// the disturbances exceeds [`mag_max_rejection_time`](Self::mag_max_rejection_time), magnetometer-based updates are performed, but with an increased
    /// time constant.
    pub mag_dist_rejection_enabled: bool,

    /// Standard deviation of the initial bias estimation uncertainty (in degrees per second).
    ///
    /// Default value: 0.5 °/s
    pub bias_sigma_init: Float,

    /// Time in which the bias estimation uncertainty increases from 0 °/s to 0.1 °/s (in seconds).
    ///
    /// This value determines the system noise assumed by the Kalman filter.
    ///
    /// Default value: 100.0 s
    pub bias_forgetting_time: Float,

    /// Maximum expected gyroscope bias (in degrees per second).
    ///
    /// This value is used to clip the bias estimate and the measurement error in the bias estimation update step. It is
    /// further used by the rest detection algorithm in order to not regard measurements with a large but constant
    /// angular rate as rest.
    ///
    /// Default value: 2.0 °/s
    pub bias_clip: Float,

    #[cfg(feature = "motion-bias-estimation")]
    /// Standard deviation of the converged bias estimation uncertainty during motion (in degrees per second).
    ///
    /// This value determines the trust on motion bias estimation updates. A small value leads to fast convergence.
    ///
    /// Default value: 0.1 °/s
    pub bias_sigma_motion: Float,

    #[cfg(feature = "motion-bias-estimation")]
    /// Forgetting factor for unobservable bias in vertical direction during motion.
    ///
    /// As magnetometer measurements are deliberately not used during motion bias estimation, gyroscope bias is not
    /// observable in vertical direction. This value is the relative weight of an artificial zero measurement that
    /// ensures that the bias estimate in the unobservable direction will eventually decay to zero.
    ///
    /// Default value: 0.0001
    pub bias_vertical_forgetting_factor: Float,

    /// Standard deviation of the converged bias estimation uncertainty during rest (in degrees per second).
    ///
    /// This value determines the trust on rest bias estimation updates. A small value leads to fast convergence.
    ///
    /// Default value: 0.03 °
    pub bias_sigma_rest: Float,

    /// Time threshold for rest detection (in seconds).
    ///
    /// Rest is detected when the measurements have been close to the low-pass filtered reference for the given time.
    ///
    /// Default value: 1.5 s
    pub rest_min_t: Float,

    /// Time constant for the low-pass filter used in rest detection (in seconds).
    ///
    /// This time constant characterizes a second-order Butterworth low-pass filter used to obtain the reference for
    /// rest detection.
    ///
    /// Default value: 0.5 s
    pub rest_filter_tau: Float,

    /// Angular velocity threshold for rest detection (in °/s).
    ///
    /// For rest to be detected, the norm of the deviation between measurement and reference must be below the given
    /// threshold. (Furthermore, the absolute value of each component must be below [`bias_clip`](Self::bias_clip)).
    ///
    /// Default value: 2.0 °/s/
    pub rest_th_gyr: Float,

    /// Acceleration threshold for rest detection (in m/s²).
    ///
    /// For rest to be detected, the norm of the deviation between measurement and reference must be below the given
    /// threshold.
    ///
    /// Default value: 0.5 m/s²
    pub rest_th_acc: Float,

    /// Time constant for current norm/dip value in magnetic disturbance detection (in seconds).
    ///
    /// This (very fast) low-pass filter is intended to provide additional robustness when the magnetometer measurements
    /// are noisy or not sampled perfectly in sync with the gyroscope measurements. Set to -1 to disable the low-pass
    /// filter and directly use the magnetometer measurements.
    ///
    /// Default value: 0.05 s
    pub mag_current_tau: Float,

    /// Time constant for the adjustment of the magnetic field reference (in seconds).
    ///
    /// This adjustment allows the reference estimate to converge to the observed undisturbed field.
    ///
    /// Default value: 20.0 s
    pub mag_ref_tau: Float,

    /// Relative threshold for the magnetic field strength for magnetic disturbance detection.
    ///
    /// This value is relative to the reference norm.
    ///
    /// Default value: 0.1 (10%)
    pub mag_norm_th: Float,

    /// Threshold for the magnetic field dip angle for magnetic disturbance detection (in degrees).
    ///
    /// Default vaule: 10 °
    pub mag_dip_th: Float,

    /// Duration after which to accept a different homogeneous magnetic field (in seconds).
    ///
    /// A different magnetic field reference is accepted as the new field when the measurements are within the thresholds
    /// [`mag_norm_th`](Self::mag_norm_th) and [`mag_dip_th`](Self::mag_dip_th) for the given time. Additionally, only phases with sufficient movement, specified by
    /// [`mag_new_min_gyr`](Self::mag_new_min_gyr), count.
    ///
    /// Default value: 20.0
    pub mag_new_time: Float,

    /// Duration after which to accept a homogeneous magnetic field for the first time (in seconds).
    ///
    /// This value is used instead of [`mag_new_time`](Self::mag_new_time) when there is no current estimate in order to allow for the initial
    /// magnetic field reference to be obtained faster.
    ///
    /// Default value: 5.0
    pub mag_new_first_time: Float,

    /// Minimum angular velocity needed in order to count time for new magnetic field acceptance (in °/s).
    ///
    /// Durations for which the angular velocity norm is below this threshold do not count towards reaching [`mag_new_time`](Self::mag_new_time).
    ///
    /// Default value: 20.0 °/s
    pub mag_new_min_gyr: Float,

    /// Minimum duration within thresholds after which to regard the field as undisturbed again (in seconds).
    ///
    /// Default value: 0.5 s
    pub mag_min_undisturbed_time: Float,

    /// Maximum duration of full magnetic disturbance rejection (in seconds).
    ///
    /// For magnetic disturbances up to this duration, heading correction is fully disabled and heading changes are
    /// tracked by gyroscope only. After this duration (or for many small disturbed phases without sufficient time in the
    /// undisturbed field in between), the heading correction is performed with an increased time constant (see
    /// [`mag_rejection_factor`](Self::mag_rejection_factor)).
    ///
    /// Default value: 60.0 s
    pub mag_max_rejection_time: Float,

    /// Factor by which to slow the heading correction during long disturbed phases.
    ///
    /// After [`mag_max_rejection_time`](Self::mag_max_rejection_time) of full magnetic disturbance rejection, heading correction is performed with an
    /// increased time constant. This parameter (approximately) specifies the factor of the increase.
    ///
    /// Furthermore, after spending [`mag_max_rejection_time`](Self::mag_max_rejection_time)/[`mag_rejection_factor`](Self::mag_rejection_factor) seconds in an undisturbed magnetic field,
    /// the time is reset and full magnetic disturbance rejection will be performed for up to [`mag_max_rejection_time`](Self::mag_max_rejection_time) again.
    ///
    /// Default value: 2.0
    pub mag_rejection_factor: Float,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            tau_acc: 3.0,
            tau_mag: 9.0,
            #[cfg(feature = "motion-bias-estimation")]
            motion_bias_est_enabled: true,
            rest_bias_est_enabled: true,
            mag_dist_rejection_enabled: true,
            bias_sigma_init: 0.5,
            bias_forgetting_time: 100.0,
            bias_clip: 2.0,
            #[cfg(feature = "motion-bias-estimation")]
            bias_sigma_motion: 0.1,
            #[cfg(feature = "motion-bias-estimation")]
            bias_vertical_forgetting_factor: 0.0001,
            bias_sigma_rest: 0.03,
            rest_min_t: 1.5,
            rest_filter_tau: 0.5,
            rest_th_gyr: 2.0,
            rest_th_acc: 0.5,
            mag_current_tau: 0.05,
            mag_ref_tau: 20.0,
            mag_norm_th: 0.1,
            mag_dip_th: 10.0,
            mag_new_time: 20.0,
            mag_new_first_time: 5.0,
            mag_new_min_gyr: 20.0,
            mag_min_undisturbed_time: 0.5,
            mag_max_rejection_time: 60.0,
            mag_rejection_factor: 2.0,
        }
    }
}

/// Struct containing the filter state of the VQF class.
///
/// The relevant parts of the state can be accessed via functions of the VQF class, e.g. [`VQF::quat_6d()`],
/// [`VQF::quat_9d()`], [`VQF::bias_estimate()`], [`VQF::set_bias_estimate()`], [`VQF::rest_detected()`] and
/// [`VQF::mag_dist_detected()`]. To reset the state to the initial values, use [`VQF::reset_state()`].
///
/// Direct access to the full state is typically not needed but can be useful in some cases, e.g. for debugging. For this
/// purpose, the state can be accessed by [`VQF::state()`] and set by [`VQF::state_mut()`].
#[cfg_attr(doc, doc = include_str!("../katex.html"))]
#[derive(Clone, Copy, Default)]
pub struct State {
    /// Angular velocity strapdown integration quaternion $^{\mathcal{S}\_i}_{\mathcal{I}\_i}\mathbf{q}$.
    pub gyr_quat: Quaternion,

    /// Inclination correction quaternion $^{\mathcal{I}\_i}\_{\mathcal{E}_i}\mathbf{q}$.
    pub acc_quat: Quaternion,

    /// Heading difference $\delta$ between $\mathcal{E}_i$ and $\mathcal{E}$.
    ///
    /// $^{\mathcal{E}\_i}\_{\mathcal{E}}\mathbf{q} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
    /// \sin\frac{\delta}{2}\end{bmatrix}^T$.
    pub delta: Float,

    /// True if it has been detected that the IMU is currently at rest.
    ///
    /// Used to switch between rest and motion gyroscope bias estimation.
    pub rest_detected: bool,

    /// True if magnetic disturbances have been detected.
    pub mag_dist_detected: bool,

    /// Last low-pass filtered acceleration in the $\mathcal{I}_i$ frame.
    pub last_acc_lp: [Float; 3],

    /// Internal low-pass filter state for [`last_acc_lp`](Self::last_acc_lp).
    pub acc_lp_state: [[f64; 2]; 3],

    /// Last inclination correction angular rate.
    ///
    /// Change to inclination correction quaternion $^{\mathcal{I}\_i}\_{\mathcal{E}_i}\mathbf{q}$ performed in the
    /// last accelerometer update, expressed as an angular rate (in rad/s).
    pub last_acc_corr_angular_rate: Float,

    /// Gain used for heading correction to ensure fast initial convergence.
    ///
    /// This value is used as the gain for heading correction in the beginning if it is larger than the normal filter
    /// gain. It is initialized to 1 and then updated to 0.5, 0.33, 0.25, ... After [`Params::tau_mag`] seconds, it is
    /// set to zero.
    pub k_mag_init: Float,

    /// Last heading disagreement angle.
    ///
    /// Disagreement between the heading $\hat\delta$ estimated from the last magnetometer sample and the state
    /// $\delta$ (in rad).
    pub last_mag_dis_angle: Float,

    /// Last heading correction angular rate.
    ///
    /// Change to heading $\delta$ performed in the last magnetometer update,
    /// expressed as an angular rate (in rad/s).
    pub last_mag_corr_angular_rate: Float,

    /// Current gyroscope bias estimate (in rad/s).
    pub bias: [Float; 3],
    #[cfg(feature = "motion-bias-estimation")]
    /// Covariance matrix of the gyroscope bias estimate.
    ///
    /// The 3x3 matrix is stored in row-major order. Note that for numeric reasons the internal unit used is 0.01 °/s,
    /// i.e. to get the standard deviation in degrees per second use $\sigma = \frac{\sqrt{p_{ii}}}{100}$.
    pub bias_p: Matrix<3, 3>,

    #[cfg(not(feature = "motion-bias-estimation"))]
    // If only rest gyr bias estimation is enabled, P and K of the KF are always diagonal
    // and matrix inversion is not needed. If motion bias estimation is disabled at compile
    // time, storing the full P matrix is not necessary.
    pub bias_p: Float,

    #[cfg(feature = "motion-bias-estimation")]
    /// Internal state of the Butterworth low-pass filter for the rotation matrix coefficients used in motion
    /// bias estimation.
    pub motion_bias_est_rlp_state: [[f64; 2]; 9],
    #[cfg(feature = "motion-bias-estimation")]
    /// Internal low-pass filter state for the rotated bias estimate used in motion bias estimation.
    pub motion_bias_est_bias_lp_state: [[f64; 2]; 2],

    /// Last (squared) deviations from the reference of the last sample used in rest detection.
    ///
    /// Looking at those values can be useful to understand how rest detection is working and which thresholds are
    /// suitable. The array contains the last values for gyroscope and accelerometer in the respective
    /// units. Note that the values are squared.
    ///
    /// The method [`VQF::relative_rest_deviations()`] provides an easier way to obtain and interpret those values.
    pub rest_last_squared_deviations: [Float; 2],

    /// The current duration for which all sensor readings are within the rest detection thresholds.
    ///
    /// Rest is detected if this value is larger or equal to [`Params::rest_min_t`].
    pub rest_t: Float,

    /// Last low-pass filtered gyroscope measurement used as the reference for rest detection.
    ///
    /// Note that this value is also used for gyroscope bias estimation when rest is detected.
    pub rest_last_gyr_lp: [Float; 3],

    /// Internal low-pass filter state for [`rest_last_gyr_lp`](Self::rest_last_gyr_lp).
    pub rest_gyr_lp_state: [[f64; 2]; 3],

    /// Last low-pass filtered accelerometer measurement used as the reference for rest detection.
    pub rest_last_acc_lp: [Float; 3],

    /// Internal low-pass filter state for [`rest_last_acc_lp`](Self::rest_last_acc_lp).
    pub rest_acc_lp_state: [[f64; 2]; 3],

    /// Norm of the currently accepted magnetic field reference.
    ///
    /// A value of -1 indicates that no homogeneous field is found yet.
    pub mag_ref_norm: Float,

    /// Dip angle of the currently accepted magnetic field reference.
    pub mag_ref_dip: Float,

    /// The current duration for which the current norm and dip are close to the reference.
    ///
    /// The magnetic field is regarded as undisturbed when this value reaches [`Params::mag_min_undisturbed_time`].
    pub mag_undisturbed_t: Float,

    /// The current duration for which the magnetic field was rejected.
    ///
    /// If the magnetic field is disturbed and this value is smaller than [`Params::mag_max_rejection_time`], heading
    /// correction updates are fully disabled.
    pub mag_reject_t: Float,

    /// Norm of the alternative magnetic field reference currently being evaluated.
    pub mag_candidate_norm: Float,

    /// Dip angle of the alternative magnetic field reference currently being evaluated.
    pub mag_candidate_dip: Float,

    /// The current duration for which the norm and dip are close to the candidate.
    ///
    /// If this value exceeds [`Params::mag_new_time`] (or [`Params::mag_new_first_time`] if [`mag_ref_norm`](Self::mag_ref_norm) < 0), the current
    /// candidate is accepted as the new reference.
    pub mag_candidate_t: Float,

    /// Norm and dip angle of the current magnetometer measurements.
    ///
    /// Slightly low-pass filtered, see [`Params::mag_current_tau`].
    pub mag_norm_dip: [Float; 2],

    /// Internal low-pass filter state for the current norm and dip angle.
    pub mag_norm_dip_lp_state: [[f64; 2]; 2],
}

/// Struct containing coefficients used by the VQF class.
///
/// Coefficients are values that depend on the parameters and the sampling times, but do not change during update steps.
/// They are calculated in [`VQF::new()`].
#[cfg_attr(doc, doc = include_str!("../katex.html"))]
#[derive(Clone, Copy, Default)]
pub struct Coefficients {
    /// Sampling time of the gyroscope measurements (in seconds).
    pub gyr_ts: Float,

    /// Sampling time of the accelerometer measurements (in seconds).
    pub acc_ts: Float,

    /// Sampling time of the magnetometer measurements (in seconds).
    pub mag_ts: Float,

    /// Numerator coefficients of the acceleration low-pass filter.
    ///
    /// The array contains $\begin{bmatrix}b_0 & b_1 & b_2\end{bmatrix}$.
    pub acc_lp_b: [f64; 3],

    /// Denominator coefficients of the acceleration low-pass filter.
    ///
    /// The array contains $\begin{bmatrix}a_1 & a_2\end{bmatrix}$ and $a_0=1$.
    pub acc_lp_a: [f64; 2],

    /// Gain of the first-order filter used for heading correction.
    pub k_mag: Float,

    /// Variance of the initial gyroscope bias estimate.
    pub bias_p0: Float,

    /// System noise variance used in gyroscope bias estimation.
    pub bias_v: Float,

    #[cfg(feature = "motion-bias-estimation")]
    /// Measurement noise variance for the motion gyroscope bias estimation update.
    pub bias_motion_w: Float,

    #[cfg(feature = "motion-bias-estimation")]
    /// Measurement noise variance for the motion gyroscope bias estimation update in vertical direction.
    pub bias_vertical_w: Float,

    /// Measurement noise variance for the rest gyroscope bias estimation update.
    pub bias_rest_w: Float,

    /// Numerator coefficients of the gyroscope measurement low-pass filter for rest detection.
    pub rest_gyr_lp_b: [f64; 3],

    /// Denominator coefficients of the gyroscope measurement low-pass filter for rest detection.
    pub rest_gyr_lp_a: [f64; 2],

    /// Numerator coefficients of the accelerometer measurement low-pass filter for rest detection.
    pub rest_acc_lp_b: [f64; 3],

    /// Denominator coefficients of the accelerometer measurement low-pass filter for rest detection.
    pub rest_acc_lp_a: [f64; 2],

    /// Gain of the first-order filter used for to update the magnetic field reference and candidate.
    pub k_mag_ref: Float,

    /// Numerator coefficients of the low-pass filter for the current magnetic norm and dip.
    pub mag_norm_dip_lp_b: [f64; 3],

    /// Denominator coefficients of the low-pass filter for the current magnetic norm and dip.
    pub mag_norm_dip_lp_a: [f64; 2],
}

/// A Versatile Quaternion-based Filter for IMU Orientation Estimation.
///
/// This class implements the orientation estimation filter described in the following publication:
/// > D. Laidig and T. Seel. "VQF: Highly Accurate IMU Orientation Estimation with Bias Estimation and Magnetic
/// > Disturbance Rejection." Information Fusion 2023, 91, 187–204.
/// > [doi:10.1016/j.inffus.2022.10.014](https://doi.org/10.1016/j.inffus.2022.10.014).
/// > [Accepted manuscript available at [arXiv:2203.17024](https://arxiv.org/abs/2203.17024).]
///
/// The filter can perform simultaneous 6D (magnetometer-free) and 9D (gyr+acc+mag) sensor fusion and can also be used
/// without magnetometer data. It performs rest detection, gyroscope bias estimation during rest and motion, and magnetic
/// disturbance detection and rejection. Different sampling rates for gyroscopes, accelerometers, and magnetometers are
/// supported as well. While in most cases, the defaults will be reasonable, the algorithm can be influenced via a
/// number of tuning parameters.
///
/// To use this implementation,
/// 1. create a instance of the class and provide the sampling time and, optionally, parameters
/// 2. for every sample, call one of the update functions to feed the algorithm with IMU data
/// 3. access the estimation results with [`quat_6d()`](Self::quat_6d()), [`quat_9d()`](Self::quat_9d()) and
/// the other getter methods.
///
/// This class is a port of the official, original C++ implementation of the algorithm. For usage in C++/Python/MATLAB,
/// the original implementations should be used.
pub struct VQF {
    params: Params,
    state: State,
    coeffs: Coefficients,
}

#[inline(always)]
fn square(t: Float) -> Float {
    t * t
}

#[inline(always)]
fn abs(t: Float) -> Float {
    #[cfg(feature = "std")]
    return t.abs();
    #[cfg(feature = "libm")]
    return Math::<Float>::fabs(t);
}

#[cfg_attr(doc, doc = include_str!("../katex.html"))]
impl VQF {
    /// Creates a new VQF instance.
    ///
    /// In the most common case (using the default parameters and all data being sampled with the same frequency,
    /// create the class like this:
    /// ```rust
    /// # use vqf_rs::VQF;
    /// let vqf = VQF::new(0.01, None, None, None); // 0.01 s sampling time, i.e. 100 Hz
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `gyr_ts`, `acc_ts`, or `mag_ts` is negative.
    pub fn new(
        gyr_ts: Float,
        acc_ts: Option<Float>,
        mag_ts: Option<Float>,
        params: Option<Params>,
    ) -> Self {
        let mut ret = Self {
            params: params.unwrap_or_default(),
            state: Default::default(),
            coeffs: Default::default(),
        };
        ret.coeffs.gyr_ts = gyr_ts;
        ret.coeffs.acc_ts = acc_ts.unwrap_or(gyr_ts);
        ret.coeffs.mag_ts = mag_ts.unwrap_or(gyr_ts);
        ret.setup();
        ret
    }

    /// Performs gyroscope update step, using the measurement in rad/s.
    ///
    /// It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
    /// different sampling rates. Otherwise, simply use [`update()`](Self::update()).
    pub fn update_gyr(&mut self, gyr: [Float; 3]) {
        // rest detection
        if self.params.rest_bias_est_enabled || self.params.mag_dist_rejection_enabled {
            Self::filter_vec(
                gyr,
                self.params.rest_filter_tau,
                self.coeffs.gyr_ts,
                self.coeffs.rest_gyr_lp_b,
                self.coeffs.rest_gyr_lp_a,
                &mut self.state.rest_gyr_lp_state,
                &mut self.state.rest_last_gyr_lp,
            );

            self.state.rest_last_squared_deviations[0] =
                square(gyr[0] - self.state.rest_last_gyr_lp[0])
                    + square(gyr[1] - self.state.rest_last_gyr_lp[1])
                    + square(gyr[2] - self.state.rest_last_gyr_lp[2]);

            let bias_clip = self.params.bias_clip * (fc::PI / 180.0);
            if self.state.rest_last_squared_deviations[0]
                >= square(self.params.rest_th_gyr * (fc::PI / 180.0))
                || abs(self.state.rest_last_gyr_lp[0]) > bias_clip
                || abs(self.state.rest_last_gyr_lp[1]) > bias_clip
                || abs(self.state.rest_last_gyr_lp[2]) > bias_clip
            {
                self.state.rest_t = 0.0;
                self.state.rest_detected = false;
            }
        }

        // remove estimated gyro bias
        let gyr_no_bias = [
            gyr[0] - self.state.bias[0],
            gyr[1] - self.state.bias[1],
            gyr[2] - self.state.bias[2],
        ];

        // gyroscope prediction step
        let gyr_norm = Self::norm(&gyr_no_bias);
        let angle = gyr_norm * self.coeffs.gyr_ts;
        if gyr_norm > Float::EPSILON {
            let c = Math::<Float>::cos(angle / 2.0);
            let s = Math::<Float>::sin(angle / 2.0) / gyr_norm;
            let gyr_step_quat = [
                c,
                s * gyr_no_bias[0],
                s * gyr_no_bias[1],
                s * gyr_no_bias[2],
            ];
            self.state.gyr_quat *= gyr_step_quat.into();
            self.state.gyr_quat.normalize();
        }
    }

    /// Performs accelerometer update step, using the measurement in m/s².
    ///
    /// It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
    /// different sampling rates. Otherwise, simply use [`update()`](Self::update()).
    ///
    /// Should be called after [`update_gyr()`](Self::update_gyr()) and before [`update_mag()`](Self::update_mag()).
    pub fn update_acc(&mut self, acc: [Float; 3]) {
        // ignore [0 0 0] samples
        if acc == [0.0; 3] {
            return;
        }

        // rest detection
        if self.params.rest_bias_est_enabled {
            Self::filter_vec(
                acc,
                self.params.rest_filter_tau,
                self.coeffs.acc_ts,
                self.coeffs.rest_acc_lp_b,
                self.coeffs.rest_acc_lp_a,
                &mut self.state.rest_acc_lp_state,
                &mut self.state.rest_last_acc_lp,
            );

            self.state.rest_last_squared_deviations[1] =
                square(acc[0] - self.state.rest_last_acc_lp[0])
                    + square(acc[1] - self.state.rest_last_acc_lp[1])
                    + square(acc[2] - self.state.rest_last_acc_lp[2]);

            if self.state.rest_last_squared_deviations[1] >= square(self.params.rest_th_acc) {
                self.state.rest_t = 0.0;
                self.state.rest_detected = false;
            } else {
                self.state.rest_t += self.coeffs.acc_ts;
                if self.state.rest_t >= self.params.rest_min_t {
                    self.state.rest_detected = true;
                }
            }
        }

        // filter acc in inertial frame
        let acc_earth = self.state.gyr_quat.rotate(acc);
        Self::filter_vec(
            acc_earth,
            self.params.tau_acc,
            self.coeffs.acc_ts,
            self.coeffs.acc_lp_b,
            self.coeffs.acc_lp_a,
            &mut self.state.acc_lp_state,
            &mut self.state.last_acc_lp,
        );

        // transform to 6D earth frame and normalize
        let mut acc_earth = self.state.acc_quat.rotate(self.state.last_acc_lp);
        Self::normalize(&mut acc_earth);

        // inclination correction
        let q_w = Math::<Float>::sqrt((acc_earth[2] + 1.0) / 2.0);
        let acc_corr_quat: Quaternion = if q_w > 1e-6 {
            [
                q_w,
                0.5 * acc_earth[1] / q_w,
                -0.5 * acc_earth[0] / q_w,
                0.0,
            ]
            .into()
        } else {
            // to avoid numeric issues when acc is close to [0 0 -1], i.e. the correction step is close (<= 0.00011°) to 180°:
            [0.0, 1.0, 0.0, 1.0].into()
        };
        self.state.acc_quat = acc_corr_quat * self.state.acc_quat;
        self.state.acc_quat.normalize();

        // calculate correction angular rate to facilitate debugging
        self.state.last_acc_corr_angular_rate =
            Math::<Float>::acos(acc_earth[2]) / self.coeffs.acc_ts;

        // bias estimation
        #[cfg(feature = "motion-bias-estimation")]
        {
            if self.params.motion_bias_est_enabled || self.params.rest_bias_est_enabled {
                let bias_clip = self.params.bias_clip * (fc::PI / 180.0);

                // get rotation matrix corresponding to accGyrQuat
                let acc_gyr_quat = self.quat_6d();
                let mut r: Matrix<3, 3> = [
                    [
                        1.0 - 2.0 * square(acc_gyr_quat.2) - 2.0 * square(acc_gyr_quat.3), // r11
                        2.0 * (acc_gyr_quat.2 * acc_gyr_quat.1 - acc_gyr_quat.0 * acc_gyr_quat.3), // r12
                        2.0 * (acc_gyr_quat.0 * acc_gyr_quat.2 + acc_gyr_quat.3 * acc_gyr_quat.1), // r13
                    ],
                    [
                        2.0 * (acc_gyr_quat.0 * acc_gyr_quat.3 + acc_gyr_quat.2 * acc_gyr_quat.1), // r21
                        1.0 - 2.0 * square(acc_gyr_quat.1) - 2.0 * square(acc_gyr_quat.3), // r22
                        2.0 * (acc_gyr_quat.2 * acc_gyr_quat.3 - acc_gyr_quat.1 * acc_gyr_quat.0), // r23
                    ],
                    [
                        2.0 * (acc_gyr_quat.3 * acc_gyr_quat.1 - acc_gyr_quat.0 * acc_gyr_quat.2), // r31
                        2.0 * (acc_gyr_quat.0 * acc_gyr_quat.1 + acc_gyr_quat.3 * acc_gyr_quat.2), // r32
                        1.0 - 2.0 * square(acc_gyr_quat.1) - 2.0 * square(acc_gyr_quat.2), // r33
                    ],
                ]
                .into();

                // calculate R*b_hat (only the x and y component, as z is not needed)
                let mut bias_lp = [
                    r.0[0][0] * self.state.bias[0]
                        + r.0[0][1] * self.state.bias[1]
                        + r.0[0][2] * self.state.bias[2],
                    r.0[1][0] * self.state.bias[0]
                        + r.0[1][1] * self.state.bias[1]
                        + r.0[1][2] * self.state.bias[2],
                ];

                // low-pass filter R and R*b_hat
                let r_arr: &mut [Float; 9] = flatten(&mut r.0);
                Self::filter_vec(
                    *r_arr,
                    self.params.tau_acc,
                    self.coeffs.acc_ts,
                    self.coeffs.acc_lp_b,
                    self.coeffs.acc_lp_a,
                    &mut self.state.motion_bias_est_rlp_state,
                    r_arr,
                );
                Self::filter_vec(
                    bias_lp,
                    self.params.tau_acc,
                    self.coeffs.acc_ts,
                    self.coeffs.acc_lp_b,
                    self.coeffs.acc_lp_a,
                    &mut self.state.motion_bias_est_bias_lp_state,
                    &mut bias_lp,
                );

                // set measurement error and covariance for the respective Kalman filter update
                let (mut e, w) = if self.state.rest_detected && self.params.rest_bias_est_enabled {
                    let e = [
                        self.state.rest_last_gyr_lp[0] - self.state.bias[0],
                        self.state.rest_last_gyr_lp[1] - self.state.bias[1],
                        self.state.rest_last_gyr_lp[2] - self.state.bias[2],
                    ];
                    r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into();
                    let w = [self.coeffs.bias_rest_w; 3];
                    (e, w)
                } else if self.params.motion_bias_est_enabled {
                    let e = [
                        -acc_earth[1] / self.coeffs.acc_ts + bias_lp[0]
                            - r.0[0][0] * self.state.bias[0]
                            - r.0[0][1] * self.state.bias[1]
                            - r.0[0][2] * self.state.bias[2],
                        acc_earth[0] / self.coeffs.acc_ts + bias_lp[1]
                            - r.0[1][0] * self.state.bias[0]
                            - r.0[1][1] * self.state.bias[1]
                            - r.0[1][2] * self.state.bias[2],
                        -r.0[2][0] * self.state.bias[0]
                            - r.0[2][1] * self.state.bias[1]
                            - r.0[2][2] * self.state.bias[2],
                    ];
                    let w = [
                        self.coeffs.bias_motion_w,
                        self.coeffs.bias_motion_w,
                        self.coeffs.bias_vertical_w,
                    ];
                    (e, w)
                } else {
                    ([0.0; 3], [-1.0; 3])
                };

                // Kalman filter update
                // step 1: P = P + V (also increase covariance if there is no measurement update!)
                if self.state.bias_p.0[0][0] < self.coeffs.bias_p0 {
                    self.state.bias_p.0[0][0] += self.coeffs.bias_v;
                }
                if self.state.bias_p.0[1][1] < self.coeffs.bias_p0 {
                    self.state.bias_p.0[1][1] += self.coeffs.bias_v;
                }
                if self.state.bias_p.0[2][2] < self.coeffs.bias_p0 {
                    self.state.bias_p.0[2][2] += self.coeffs.bias_v;
                }

                if w[0] >= 0.0 {
                    // clip disagreement to -2..2 °/s
                    // (this also effectively limits the harm done by the first inclination correction step)
                    Self::clip(&mut e, -bias_clip, bias_clip);

                    // step 2: K = P R^T inv(W + R P R^T)
                    let mut k = self.state.bias_p * r.transpose(); // K = P R^T
                    k = r * k; // K = R P R^T
                    k.0[0][0] += w[0];
                    k.0[1][1] += w[1];
                    k.0[2][2] += w[2]; // K = W + R P R^T
                    k = k.invert().unwrap(); // K = inv(W + R P R^T)
                    k = r.transpose() * k; // K = R^T inv(W + R P R^T)
                    k = self.state.bias_p * k; // K = P R^T inv(W + R P R^T)

                    // step 3: bias = bias + K (y - R bias) = bias + K e
                    self.state.bias[0] += k.0[0][0] * e[0] + k.0[0][1] * e[1] + k.0[0][2] * e[2];
                    self.state.bias[1] += k.0[1][0] * e[0] + k.0[1][1] * e[1] + k.0[1][2] * e[2];
                    self.state.bias[2] += k.0[2][0] * e[0] + k.0[2][1] * e[1] + k.0[2][2] * e[2];

                    // step 4: P = P - K R P
                    k *= r; // K = K R
                    k *= self.state.bias_p; // K = K R P
                    self.state.bias_p -= k;

                    // clip bias estimate to -2..2 °/s
                    Self::clip(&mut self.state.bias, -bias_clip, bias_clip);
                }
            }
        }
        #[cfg(not(feature = "motion-bias-estimation"))]
        {
            // simplified implementation of bias estimation for the special case in which only rest bias estimation is enabled
            if self.params.rest_bias_est_enabled {
                let bias_clip = self.params.bias_clip * (fc::PI / 180.0);
                if self.state.bias_p < self.coeffs.bias_p0 {
                    self.state.bias_p += self.coeffs.bias_v;
                }
                if self.state.rest_detected {
                    let mut e = [
                        self.state.rest_last_gyr_lp[0] - self.state.bias[0],
                        self.state.rest_last_gyr_lp[1] - self.state.bias[1],
                        self.state.rest_last_gyr_lp[2] - self.state.bias[2],
                    ];
                    Self::clip(&mut e, -bias_clip, bias_clip);

                    // Kalman filter update, simplified scalar version for rest update
                    // (this version only uses the first entry of P as P is diagonal and all diagonal elements are the same)
                    // step 1: P = P + V (done above!)
                    // step 2: K = P R^T inv(W + R P R^T)
                    let k = self.state.bias_p / (self.coeffs.bias_rest_w + self.state.bias_p);
                    // step 3: bias = bias + K (y - R bias) = bias + K e
                    self.state.bias[0] += k * e[0];
                    self.state.bias[1] += k * e[1];
                    self.state.bias[2] += k * e[2];
                    // step 4: P = P - K R P
                    self.state.bias_p -= k * self.state.bias_p;
                    Self::clip(&mut self.state.bias, -bias_clip, bias_clip);
                }
            }
        }
    }

    /// Performs magnetometer update step.
    ///
    /// It is only necessary to call this function directly if gyroscope, accelerometers and magnetometers have
    /// different sampling rates. Otherwise, simply use [`update()`](Self::update()).
    ///
    /// Should be called after [`update_acc()`](Self::update_acc()).
    pub fn update_mag(&mut self, mag: [Float; 3]) {
        // ignore [0 0 0] samples
        if mag == [0.0; 3] {
            return;
        }

        // bring magnetometer measurement into 6D earth frame
        let acc_gyr_quat = self.quat_6d();
        let mag_earth = acc_gyr_quat.rotate(mag);

        if self.params.mag_dist_rejection_enabled {
            self.state.mag_norm_dip[0] = Self::norm(&mag_earth);
            self.state.mag_norm_dip[1] =
                -Math::<Float>::asin(mag_earth[2] / self.state.mag_norm_dip[0]);

            if self.params.mag_current_tau > 0.0 {
                Self::filter_vec(
                    self.state.mag_norm_dip,
                    self.params.mag_current_tau,
                    self.coeffs.mag_ts,
                    self.coeffs.mag_norm_dip_lp_b,
                    self.coeffs.mag_norm_dip_lp_a,
                    &mut self.state.mag_norm_dip_lp_state,
                    &mut self.state.mag_norm_dip,
                );
            }

            // magnetic disturbance detection
            if abs(self.state.mag_norm_dip[0] - self.state.mag_ref_norm)
                < self.params.mag_norm_th * self.state.mag_ref_norm
                && abs(self.state.mag_norm_dip[1] - self.state.mag_ref_dip)
                    < self.params.mag_dip_th * (fc::PI / 180.0)
            {
                self.state.mag_undisturbed_t += self.coeffs.mag_ts;
                if self.state.mag_undisturbed_t >= self.params.mag_min_undisturbed_time {
                    self.state.mag_dist_detected = false;
                    self.state.mag_ref_norm += self.coeffs.k_mag_ref
                        * (self.state.mag_norm_dip[0] - self.state.mag_ref_norm);
                    self.state.mag_ref_dip += self.coeffs.k_mag_ref
                        * (self.state.mag_norm_dip[1] - self.state.mag_ref_dip);
                }
            } else {
                self.state.mag_undisturbed_t = 0.0;
                self.state.mag_dist_detected = true;
            }

            // new magnetic field acceptance
            if abs(self.state.mag_norm_dip[0] - self.state.mag_candidate_norm)
                < self.params.mag_norm_th * self.state.mag_candidate_norm
                && abs(self.state.mag_norm_dip[1] - self.state.mag_candidate_dip)
                    < self.params.mag_dip_th * (fc::PI / 180.0)
            {
                if Self::norm(&self.state.rest_last_gyr_lp)
                    >= self.params.mag_new_min_gyr * fc::PI / 180.0
                {
                    self.state.mag_candidate_t += self.coeffs.mag_ts;
                }
                self.state.mag_candidate_norm += self.coeffs.k_mag_ref
                    * (self.state.mag_norm_dip[0] - self.state.mag_candidate_norm);
                self.state.mag_candidate_dip += self.coeffs.k_mag_ref
                    * (self.state.mag_norm_dip[1] - self.state.mag_candidate_dip);

                if self.state.mag_dist_detected
                    && (self.state.mag_candidate_t >= self.params.mag_new_time
                        || (self.state.mag_ref_norm == 0.0
                            && self.state.mag_candidate_t >= self.params.mag_new_first_time))
                {
                    self.state.mag_ref_norm = self.state.mag_candidate_norm;
                    self.state.mag_ref_dip = self.state.mag_candidate_dip;
                    self.state.mag_dist_detected = false;
                    self.state.mag_undisturbed_t = self.params.mag_min_undisturbed_time;
                }
            } else {
                self.state.mag_candidate_t = 0.0;
                self.state.mag_candidate_norm = self.state.mag_norm_dip[0];
                self.state.mag_candidate_dip = self.state.mag_norm_dip[1];
            }
        }

        // calculate disagreement angle based on current magnetometer measurement
        self.state.last_mag_dis_angle =
            Math::<Float>::atan2(mag_earth[0], mag_earth[1]) - self.state.delta;

        // make sure the disagreement angle is in the range [-pi, pi]
        if self.state.last_mag_dis_angle > fc::PI {
            self.state.last_mag_dis_angle -= 2.0 * fc::PI;
        } else if self.state.last_mag_dis_angle < (-fc::PI) {
            self.state.last_mag_dis_angle += 2.0 * fc::PI;
        }

        let mut k = self.coeffs.k_mag;

        if self.params.mag_dist_rejection_enabled {
            // magnetic disturbance rejection
            if self.state.mag_dist_detected {
                if self.state.mag_reject_t <= self.params.mag_max_rejection_time {
                    self.state.mag_reject_t += self.coeffs.mag_ts;
                    k = 0.0;
                } else {
                    k /= self.params.mag_rejection_factor;
                }
            } else {
                self.state.mag_reject_t = (0.0 as Float).max(
                    self.state.mag_reject_t - self.params.mag_rejection_factor * self.coeffs.mag_ts,
                );
            }
        }

        // ensure fast initial convergence
        if self.state.k_mag_init != 0.0 {
            // make sure that the gain k is at least 1/N, N=1,2,3,... in the first few samples
            if k < self.state.k_mag_init {
                k = self.state.k_mag_init;
            }

            // iterative expression to calculate 1/N
            self.state.k_mag_init = self.state.k_mag_init / (self.state.k_mag_init + 1.0);

            // disable if t > tauMag
            if self.state.k_mag_init * self.params.tau_mag < self.coeffs.mag_ts {
                self.state.k_mag_init = 0.0;
            }
        }

        // first-order filter step
        self.state.delta += k * self.state.last_mag_dis_angle;
        // calculate correction angular rate to facilitate debugging
        self.state.last_mag_corr_angular_rate =
            k * self.state.last_mag_dis_angle / self.coeffs.mag_ts;

        // make sure delta is in the range [-pi, pi]
        if self.state.delta > fc::PI {
            self.state.delta -= 2.0 * fc::PI;
        } else if self.state.delta < -fc::PI {
            self.state.delta += 2.0 * fc::PI;
        }
    }

    /// Performs filter update step for one sample.
    pub fn update(&mut self, gyr: [Float; 3], acc: [Float; 3], mag: Option<[Float; 3]>) {
        self.update_gyr(gyr);
        self.update_acc(acc);
        if let Some(mag) = mag {
            self.update_mag(mag);
        }
    }

    /// Returns the angular velocity strapdown integration quaternion
    /// $^{\mathcal{S}\_i}\_{\mathcal{I}_i}\mathbf{q}$.
    pub fn quat_3d(&self) -> Quaternion {
        self.state.gyr_quat
    }

    /// Returns the 6D (magnetometer-free) orientation quaternion
    /// $^{\mathcal{S}\_i}\_{\mathcal{E}_i}\mathbf{q}$.
    pub fn quat_6d(&self) -> Quaternion {
        self.state.acc_quat * self.state.gyr_quat
    }

    /// Returns the 9D (with magnetometers) orientation quaternion
    /// $^{\mathcal{S}\_i}\_{\mathcal{E}}\mathbf{q}$.
    pub fn quat_9d(&self) -> Quaternion {
        (self.state.acc_quat * self.state.gyr_quat).apply_delta(self.state.delta)
    }

    /// Returns the heading difference $\delta$ between $\mathcal{E}_i$ and $\mathcal{E}$ in radians.
    ///
    /// $^{\mathcal{E}\_i}\_{\mathcal{E}}\mathbf{q} = \begin{bmatrix}\cos\frac{\delta}{2} & 0 & 0 &
    /// \sin\frac{\delta}{2}\end{bmatrix}^T$.
    pub fn delta(&self) -> Float {
        self.state.delta
    }

    #[cfg(feature = "motion-bias-estimation")]
    /// Returns the current gyroscope bias estimate and the uncertainty in rad/s.
    ///
    /// The returned standard deviation sigma represents the estimation uncertainty in the worst direction and is based
    /// on an upper bound of the largest eigenvalue of the covariance matrix.
    pub fn bias_estimate(&self) -> ([Float; 3], Float) {
        // use largest absolute row sum as upper bound estimate for largest eigenvalue (Gershgorin circle theorem)
        // and clip output to biasSigmaInit
        let sum1 = abs(self.state.bias_p.0[0][0])
            + abs(self.state.bias_p.0[0][1])
            + abs(self.state.bias_p.0[0][2]);
        let sum2 = abs(self.state.bias_p.0[1][0])
            + abs(self.state.bias_p.0[1][1])
            + abs(self.state.bias_p.0[1][2]);
        let sum3 = abs(self.state.bias_p.0[2][0])
            + abs(self.state.bias_p.0[2][1])
            + abs(self.state.bias_p.0[2][2]);
        let p = sum1.max(sum2).max(sum3).min(self.coeffs.bias_p0);
        (
            self.state.bias,
            Math::<Float>::sqrt(p) * (fc::PI / 100.0 / 180.0),
        )
    }

    #[cfg(not(feature = "motion-bias-estimation"))]
    /// Returns the current gyroscope bias estimate and the uncertainty in rad/s.
    ///
    /// The returned standard deviation sigma represents the estimation uncertainty in the worst direction and is based
    /// on an upper bound of the largest eigenvalue of the covariance matrix.
    pub fn bias_estimate(&self) -> ([Float; 3], Float) {
        (
            self.state.bias,
            Math::<Float>::sqrt(self.state.bias_p) * (fc::PI / 100.0 / 180.0),
        )
    }

    /// Sets the current gyroscope bias estimate and the uncertainty in rad/s.
    ///
    /// If a value for the uncertainty sigma is given in `sigma`, the covariance matrix is set to a corresponding scaled
    /// identity matrix.
    pub fn set_bias_estimate(&mut self, bias: [Float; 3], sigma: Option<Float>) {
        self.state.bias = bias;
        if let Some(sigma) = sigma {
            let p = square(sigma * (180.0 * 100.0 / fc::PI));
            #[cfg(feature = "motion-bias-estimation")]
            {
                self.state.bias_p = [[p, 0.0, 0.0], [0.0, p, 0.0], [0.0, 0.0, p]].into();
            }
            #[cfg(not(feature = "motion-bias-estimation"))]
            {
                self.state.bias_p = p;
            }
        }
    }

    /// Returns true if rest was detected.
    pub fn rest_detected(&self) -> bool {
        self.state.rest_detected
    }

    /// Returns true if a disturbed magnetic field was detected.
    pub fn mag_dist_detected(&self) -> bool {
        self.state.mag_dist_detected
    }

    /// Returns the relative deviations used in rest detection.
    ///
    /// Looking at those values can be useful to understand how rest detection is working and which thresholds are
    /// suitable. The output array is filled with the last values for gyroscope and accelerometer,
    /// relative to the threshold. In order for rest to be detected, both values must stay below 1.
    pub fn relative_rest_deviations(&self) -> [Float; 2] {
        [
            Math::<Float>::sqrt(self.state.rest_last_squared_deviations[0])
                / (self.params.rest_th_gyr * (fc::PI / 180.0)),
            Math::<Float>::sqrt(self.state.rest_last_squared_deviations[1])
                / self.params.rest_th_acc,
        ]
    }

    /// Returns the norm of the currently accepted magnetic field reference.
    pub fn mag_ref_norm(&self) -> Float {
        self.state.mag_ref_norm
    }

    /// Returns the dip angle of the currently accepted magnetic field reference.
    pub fn mag_ref_dip(&self) -> Float {
        self.state.mag_ref_dip
    }

    /// Overwrites the current magnetic field reference.
    pub fn set_mag_ref(&mut self, norm: Float, dip: Float) {
        self.state.mag_ref_norm = norm;
        self.state.mag_ref_dip = dip;
    }

    /// Sets the time constant $\tau_\mathrm{acc}$ in seconds for accelerometer low-pass filtering.
    ///
    /// For more details, see [`Params::tau_acc`].
    pub fn set_tau_acc(&mut self, tau_acc: Float) {
        if self.params.tau_acc == tau_acc {
            return;
        }
        self.params.tau_acc = tau_acc;
        let mut new_b = [0.0; 3];
        let mut new_a = [0.0; 2];

        Self::filter_coeffs(
            self.params.tau_acc,
            self.coeffs.acc_ts,
            &mut new_b,
            &mut new_a,
        );
        Self::filter_adapt_state_for_coeff_change(
            self.state.last_acc_lp,
            self.coeffs.acc_lp_b,
            self.coeffs.acc_lp_a,
            new_b,
            new_a,
            &mut self.state.acc_lp_state,
        );

        #[cfg(feature = "motion-bias-estimation")]
        {
            // For R and biasLP, the last value is not saved in the state.
            // Since b0 is small (at reasonable settings), the last output is close to state[0].
            let mut r: [Float; 9] = [0.0; 9];
            for (i, val) in r.iter_mut().enumerate() {
                *val = self.state.motion_bias_est_rlp_state[i][0] as Float;
            }
            Self::filter_adapt_state_for_coeff_change(
                r,
                self.coeffs.acc_lp_b,
                self.coeffs.acc_lp_a,
                new_b,
                new_a,
                &mut self.state.motion_bias_est_rlp_state,
            );
            let mut bias_lp: [Float; 2] = [0.0; 2];
            for (i, val) in bias_lp.iter_mut().enumerate() {
                *val = self.state.motion_bias_est_bias_lp_state[i][0] as Float;
            }
            Self::filter_adapt_state_for_coeff_change(
                bias_lp,
                self.coeffs.acc_lp_b,
                self.coeffs.acc_lp_a,
                new_b,
                new_a,
                &mut self.state.motion_bias_est_bias_lp_state,
            );
        }

        self.coeffs.acc_lp_b = new_b;
        self.coeffs.acc_lp_a = new_a;
    }

    /// Sets the time constant $\tau_\mathrm{mag}$ in seconds for the magnetometer update.
    ///
    /// For more details, see [`Params::tau_mag`].
    pub fn set_tau_mag(&mut self, tau_mag: Float) {
        self.params.tau_mag = tau_mag;
        self.coeffs.k_mag = Self::gain_from_tau(self.params.tau_mag, self.coeffs.mag_ts)
    }

    #[cfg(feature = "motion-bias-estimation")]
    /// Enables/disables gyroscope bias estimation during motion.
    pub fn set_motion_bias_est_enabled(&mut self, enabled: bool) {
        if self.params.motion_bias_est_enabled == enabled {
            return;
        }
        self.params.motion_bias_est_enabled = enabled;
        self.state.motion_bias_est_rlp_state = [[f64::NAN; 2]; 9];
        self.state.motion_bias_est_bias_lp_state = [[f64::NAN; 2]; 2];
    }

    /// Enables/disables rest detection and bias estimation during rest.
    pub fn set_rest_bias_est_enabled(&mut self, enabled: bool) {
        if self.params.rest_bias_est_enabled == enabled {
            return;
        }
        self.params.rest_bias_est_enabled = enabled;
        self.state.rest_detected = false;
        self.state.rest_last_squared_deviations = [0.0; 2];
        self.state.rest_t = 0.0;
        self.state.rest_last_gyr_lp = [0.0; 3];
        self.state.rest_gyr_lp_state = [[f64::NAN; 2]; 3];
        self.state.rest_last_acc_lp = [0.0; 3];
        self.state.rest_acc_lp_state = [[f64::NAN; 2]; 3];
    }

    /// Enables/disables magnetic disturbance detection and rejection.
    pub fn set_mag_dist_rejection_enabled(&mut self, enabled: bool) {
        if self.params.mag_dist_rejection_enabled == enabled {
            return;
        }
        self.params.mag_dist_rejection_enabled = enabled;
        self.state.mag_dist_detected = true;
        self.state.mag_ref_norm = 0.0;
        self.state.mag_ref_dip = 0.0;
        self.state.mag_undisturbed_t = 0.0;
        self.state.mag_reject_t = self.params.mag_max_rejection_time;
        self.state.mag_candidate_norm = -1.0;
        self.state.mag_candidate_dip = 0.0;
        self.state.mag_candidate_t = 0.0;
        self.state.mag_norm_dip_lp_state = [[f64::NAN; 2]; 2];
    }

    /// Sets the current thresholds for rest detection.
    ///
    /// For details about the parameters, see [`Params::rest_th_gyr`] and [`Params::rest_th_acc`].
    pub fn set_rest_detection_thresholds(&mut self, th_gyr: Float, th_acc: Float) {
        self.params.rest_th_gyr = th_gyr;
        self.params.rest_th_acc = th_acc;
    }

    /// Returns the current parameters.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Returns the coefficients used by the algorithm.
    pub fn coeffs(&self) -> &Coefficients {
        &self.coeffs
    }

    /// Returns the current state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Gets the current state for modification.
    ///
    /// This method allows to set a completely arbitrary filter state and is intended for debugging purposes.
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Resets the state to the default values at initialization.
    ///
    /// Resetting the state is equivalent to creating a new instance of this struct.
    pub fn reset_state(&mut self) {
        self.state.gyr_quat = [1.0, 0.0, 0.0, 0.0].into();
        self.state.acc_quat = [1.0, 0.0, 0.0, 0.0].into();
        self.state.delta = 0.0;

        self.state.rest_detected = false;
        self.state.mag_dist_detected = true;

        self.state.last_acc_lp = [0.0; 3];
        self.state.acc_lp_state = [[f64::NAN; 2]; 3];
        self.state.last_acc_corr_angular_rate = 0.0;

        self.state.k_mag_init = 1.0;
        self.state.last_mag_dis_angle = 0.0;
        self.state.last_mag_corr_angular_rate = 0.0;

        self.state.bias = [0.0; 3];

        #[cfg(feature = "motion-bias-estimation")]
        {
            self.state.bias_p = [
                [self.coeffs.bias_p0, 0.0, 0.0],
                [0.0, self.coeffs.bias_p0, 0.0],
                [0.0, 0.0, self.coeffs.bias_p0],
            ]
            .into();
        }

        #[cfg(not(feature = "motion-bias-estimation"))]
        {
            self.state.bias_p = self.coeffs.bias_p0;
        }

        #[cfg(feature = "motion-bias-estimation")]
        {
            self.state.motion_bias_est_rlp_state = [[f64::NAN; 2]; 9];
            self.state.motion_bias_est_bias_lp_state = [[f64::NAN; 2]; 2];
        }

        self.state.rest_last_squared_deviations = [0.0; 2];
        self.state.rest_t = 0.0;
        self.state.rest_last_gyr_lp = [0.0; 3];
        self.state.rest_gyr_lp_state = [[f64::NAN; 2]; 3];
        self.state.rest_last_acc_lp = [0.0; 3];
        self.state.rest_acc_lp_state = [[f64::NAN; 2]; 3];

        self.state.mag_ref_norm = 0.0;
        self.state.mag_ref_dip = 0.0;
        self.state.mag_undisturbed_t = 0.0;
        self.state.mag_reject_t = self.params.mag_max_rejection_time;
        self.state.mag_candidate_norm = -1.0;
        self.state.mag_candidate_dip = 0.0;
        self.state.mag_candidate_t = 0.0;
        self.state.mag_norm_dip = [0.0; 2];
        self.state.mag_norm_dip_lp_state = [[f64::NAN; 2]; 2];
    }

    fn norm<const N: usize>(vec: &[Float; N]) -> Float {
        let mut s = 0.0;
        for i in vec {
            s += i * i;
        }
        Math::<Float>::sqrt(s)
    }

    fn normalize<const N: usize>(vec: &mut [Float; N]) {
        let n = Self::norm(vec);
        if n < Float::EPSILON {
            return;
        }
        for i in vec.iter_mut() {
            *i /= n;
        }
    }

    fn clip<const N: usize>(vec: &mut [Float; N], min: Float, max: Float) {
        for i in vec.iter_mut() {
            if *i < min {
                *i = min;
            } else if *i > max {
                *i = max;
            }
        }
    }

    fn gain_from_tau(tau: Float, ts: Float) -> Float {
        assert!(ts > 0.0);
        if tau < 0.0 {
            0.0 // k=0 for negative tau (disable update)
        } else if tau == 0.0 {
            1.0 // k=1 for tau=0
        } else {
            1.0 - Math::<Float>::exp(-ts / tau) // fc = 1/(2*pi*tau)
        }
    }

    fn filter_coeffs(tau: Float, ts: Float, out_b: &mut [f64; 3], out_a: &mut [f64; 2]) {
        assert!(tau > 0.0);
        assert!(ts > 0.0);
        // second order Butterworth filter based on https://stackoverflow.com/a/52764064
        let fc = (f64c::SQRT_2 / (2.0 * f64c::PI)) / (tau as f64); // time constant of dampened, non-oscillating part of step response
        let c = Math::<f64>::tan(f64c::PI * fc * (ts as f64));
        let d = c * c + f64c::SQRT_2 * c + 1.0;
        let b0 = c * c / d;
        out_b[0] = b0;
        out_b[1] = 2.0 * b0;
        out_b[2] = b0;
        // a0 = 1.0
        out_a[0] = 2.0 * (c * c - 1.0) / d; // a1
        out_a[1] = (1.0 - f64c::SQRT_2 * c + c * c) / d; // a2
    }

    fn filter_initial_state(x0: Float, b: [f64; 3], a: [f64; 2], out: &mut [f64; 2]) {
        // initial state for steady state (equivalent to scipy.signal.lfilter_zi, obtained by setting y=x=x0 in the filter
        // update equation)
        out[0] = (x0 as f64) * (1.0 - b[0]);
        out[1] = (x0 as f64) * (b[2] - a[1]);
    }

    fn filter_adapt_state_for_coeff_change<const N: usize>(
        last_y: [Float; N],
        b_old: [f64; 3],
        a_old: [f64; 2],
        b_new: [f64; 3],
        a_new: [f64; 2],
        state: &mut [[f64; 2]; N],
    ) {
        if state[0][0].is_nan() {
            return;
        }
        for (i, row) in state.iter_mut().enumerate() {
            row[0] += (b_old[0] - b_new[0]) * last_y[i] as f64;
            row[1] += (b_old[1] - b_new[1] - a_old[0] + a_new[0]) * last_y[i] as f64;
        }
    }

    fn filter_step(x: Float, b: [f64; 3], a: [f64; 2], state: &mut [f64; 2]) -> Float {
        // difference equations based on scipy.signal.lfilter documentation
        // assumes that a0 == 1.0
        let y = b[0] * (x as f64) + state[0];
        state[0] = b[1] * (x as f64) - a[0] * y + state[1];
        state[1] = b[2] * (x as f64) - a[1] * y;
        y as Float
    }

    fn filter_vec<const N: usize>(
        x: [Float; N],
        tau: Float,
        ts: Float,
        b: [f64; 3],
        a: [f64; 2],
        state: &mut [[f64; 2]; N],
        out: &mut [Float; N],
    ) {
        assert!(N >= 2);

        // to avoid depending on a single sample, average the first samples (for duration tau)
        // and then use this average to calculate the filter initial state
        if state[0][0].is_nan() {
            // initialization phase
            if state[1][0].is_nan() {
                // first sample
                state[1][0] = 0.0; // state[1][0] is used to store the sample count
                for row in state.iter_mut() {
                    row[1] = 0.0; // state[i][1] is used to store the sum
                }
            }
            state[1][0] += 1.0;
            for i in 0..N {
                state[i][1] += x[i] as f64;
                out[i] = (state[i][1] / state[1][0]) as Float;
            }
            if (state[1][0] as Float) * ts >= tau {
                for i in 0..N {
                    Self::filter_initial_state(out[i], b, a, &mut state[i]);
                }
            }
            return;
        }

        for i in 0..N {
            out[i] = Self::filter_step(x[i], b, a, &mut state[i]);
        }
    }

    fn setup(&mut self) {
        assert!(self.coeffs.gyr_ts > 0.0);
        assert!(self.coeffs.acc_ts > 0.0);
        assert!(self.coeffs.mag_ts > 0.0);

        Self::filter_coeffs(
            self.params.tau_acc,
            self.coeffs.acc_ts,
            &mut self.coeffs.acc_lp_b,
            &mut self.coeffs.acc_lp_a,
        );

        self.coeffs.k_mag = Self::gain_from_tau(self.params.tau_mag, self.coeffs.mag_ts);

        self.coeffs.bias_p0 = square(self.params.bias_sigma_init * 100.0);
        // the system noise increases the variance from 0 to (0.1 °/s)^2 in biasForgettingTime seconds
        self.coeffs.bias_v =
            square(0.1 * 100.0) * self.coeffs.acc_ts / self.params.bias_forgetting_time;

        #[cfg(feature = "motion-bias-estimation")]
        {
            let p_motion = square(self.params.bias_sigma_motion * 100.0);
            self.coeffs.bias_motion_w = square(p_motion) / self.coeffs.bias_v + p_motion;
            self.coeffs.bias_vertical_w =
                self.coeffs.bias_motion_w / self.params.bias_vertical_forgetting_factor.max(1e-10);
        }

        let p_rest = square(self.params.bias_sigma_rest * 100.0);
        self.coeffs.bias_rest_w = square(p_rest) / self.coeffs.bias_v + p_rest;

        Self::filter_coeffs(
            self.params.rest_filter_tau,
            self.coeffs.gyr_ts,
            &mut self.coeffs.rest_gyr_lp_b,
            &mut self.coeffs.rest_gyr_lp_a,
        );
        Self::filter_coeffs(
            self.params.rest_filter_tau,
            self.coeffs.acc_ts,
            &mut self.coeffs.rest_acc_lp_b,
            &mut self.coeffs.rest_acc_lp_a,
        );

        self.coeffs.k_mag_ref = Self::gain_from_tau(self.params.mag_ref_tau, self.coeffs.mag_ts);
        if self.params.mag_current_tau > 0.0 {
            Self::filter_coeffs(
                self.params.mag_current_tau,
                self.coeffs.mag_ts,
                &mut self.coeffs.mag_norm_dip_lp_b,
                &mut self.coeffs.mag_norm_dip_lp_a,
            );
        } else {
            self.coeffs.mag_norm_dip_lp_b = [f64::NAN; 3];
            self.coeffs.mag_norm_dip_lp_a = [f64::NAN; 2];
        }

        self.reset_state();
    }
}

#[cfg(test)]
mod tests {
    use crate::{Params, Quaternion, VQF};

    #[test]
    fn basic_parity() {
        for mode in 0..=5 {
            let params = match mode {
                0 => Default::default(),
                1 => Params {
                    mag_dist_rejection_enabled: false,
                    ..Default::default()
                },
                2 => Params {
                    rest_bias_est_enabled: false,
                    ..Default::default()
                },
                3 => Params {
                    motion_bias_est_enabled: false,
                    ..Default::default()
                },
                4 => Params {
                    rest_bias_est_enabled: false,
                    motion_bias_est_enabled: false,
                    ..Default::default()
                },
                5 => Params {
                    mag_dist_rejection_enabled: false,
                    rest_bias_est_enabled: false,
                    motion_bias_est_enabled: false,
                    ..Default::default()
                },
                _ => panic!(),
            };
            let expected: Quaternion = match mode {
                0 => [0.499988, 0.499988, 0.500012, 0.500012].into(),
                1 => [0.5, 0.5, 0.5, 0.5].into(),
                2 => [0.451372, 0.453052, 0.543672, 0.543533].into(),
                3 => [0.499988, 0.499988, 0.500012, 0.500012].into(),
                4 => [0.424513, 0.454375, 0.555264, 0.55228].into(),
                5 => [0.44869, 0.478654, 0.534476, 0.532825].into(),
                _ => panic!(),
            };
            let mut vqf = VQF::new(0.01, None, None, Some(params));

            let gyr = [0.01; 3];
            let acc = [0.0, 9.8, 0.0];
            let mag = [0.5, 0.8, 0.0];

            for _ in 0..10000 {
                vqf.update(gyr, acc, Some(mag))
            }

            let quat = vqf.quat_9d();
            assert!((quat.0 - expected.0).abs() < 1e-6);
            assert!((quat.1 - expected.1).abs() < 1e-6);
            assert!((quat.2 - expected.2).abs() < 1e-6);
            assert!((quat.3 - expected.3).abs() < 1e-6);
        }
    }

    #[allow(clippy::approx_constant)]
    #[allow(non_snake_case)]
    mod wigwagwent_tests {
        use crate::{Params, VQF};

    
        #[test]
        fn single_same_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            vqf.update_gyr(gyr);
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - 1.0).abs() < 1e-6);
            assert!((quat.1 - 0.000105).abs() < 1e-6);
            assert!((quat.2 - 0.000105).abs() < 1e-6);
            assert!((quat.3 - 0.000105).abs() < 1e-6);
        }
    
        #[test]
        fn single_x_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.25, 0.0, 0.0];
            vqf.update_gyr(gyr);
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - 0.9999999).abs() < 1e-6);
            assert!((quat.1 - 0.00125).abs() < 1e-6);
            assert!((quat.2 - 0.0).abs() < 1e-6);
            assert!((quat.3 - 0.0).abs() < 1e-6);
        }
    
        #[test]
        fn single_y_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.0, 0.25, 0.0];
            vqf.update_gyr(gyr);
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - 0.9999999).abs() < 1e-6);
            assert!((quat.1 - 0.0).abs() < 1e-6);
            assert!((quat.2 - 0.00125).abs() < 1e-6);
            assert!((quat.3 - 0.0).abs() < 1e-6);
        }
    
        #[test]
        fn single_z_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.0, 0.0, 0.25];
            vqf.update_gyr(gyr);
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - 0.9999999).abs() < 1e-6);
            assert!((quat.1 - 0.0).abs() < 1e-6);
            assert!((quat.2 - 0.0).abs() < 1e-6);
            assert!((quat.3 - 0.00125).abs() < 1e-6);
        }
    
        #[test]
        fn single_different_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.054, 0.012, -0.9];
            vqf.update_gyr(gyr);
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - 0.99999).abs() < 1e-6);
            assert!((quat.1 - 0.000269999).abs() < 1e-6);
            assert!((quat.2 - 5.99998e-5).abs() < 1e-6);
            assert!((quat.3 - -0.00449998).abs() < 1e-6);
        }
    
        #[test]
        fn many_same_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            for _ in 0..10_000 {
                vqf.update_gyr(gyr);
            }
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - -0.245327).abs() < 1e-6); //slightly different results
            assert!((quat.1 - 0.559707).abs() < 1e-6);
            assert!((quat.2 - 0.559707).abs() < 1e-6);
            assert!((quat.3 - 0.559707).abs() < 1e-6);
        }
    
        #[test]
        fn many_different_3D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.054, 0.012, -0.09];
            for _ in 0..10_000 {
                vqf.update_gyr(gyr);
            }
    
            let quat = vqf.quat_3d();
            assert!((quat.0 - 0.539342).abs() < 1e-6); //slightly different results
            assert!((quat.1 - -0.430446).abs() < 1e-6);
            assert!((quat.2 - -0.0956546).abs() < 1e-6);
            assert!((quat.3 - 0.71741).abs() < 1e-6);
        }
    
        #[test]
        fn single_same_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            vqf.update_gyr(gyr);
            vqf.update_acc(acc);
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.888074).abs() < 1e-6);
            assert!((quat.1 - 0.325117).abs() < 1e-6);
            assert!((quat.2 - -0.324998).abs() < 1e-6);
            assert!((quat.3 - 0.00016151).abs() < 1e-6);
        }
    
        #[test]
        fn single_x_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [9.81, 0.0, 0.0];
            vqf.update(gyr, acc, None);
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.707107).abs() < 1e-6);
            assert!((quat.1 - 0.000148508).abs() < 1e-6);
            assert!((quat.2 - -0.707107).abs() < 1e-6);
            assert!((quat.3 - 0.000148508).abs() < 1e-6);
        }
    
        #[test]
        fn single_y_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [0.0, 9.81, 0.0];
            vqf.update(gyr, acc, None);
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.707107).abs() < 1e-6);
            assert!((quat.1 - 0.707107).abs() < 1e-6);
            assert!((quat.2 - 0.000148477).abs() < 1e-6);
            assert!((quat.3 - 0.000148477).abs() < 1e-6);
        }
    
        #[test]
        fn single_z_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [0.0, 0.0, 9.81];
            vqf.update(gyr, acc, None);
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 1.0).abs() < 1e-6);
            assert!((quat.1 - -1.72732e-20).abs() < 1e-6);
            assert!((quat.2 - -4.06576e-20).abs() < 1e-6);
            assert!((quat.3 - 0.000105).abs() < 1e-6);
        }
    
        #[test]
        fn single_different_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [4.5, 6.7, 3.2];
            vqf.update(gyr, acc, None);
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.827216).abs() < 1e-6);
            assert!((quat.1 - 0.466506).abs() < 1e-6);
            assert!((quat.2 - -0.313187).abs() < 1e-6);
            assert!((quat.3 - 0.000168725).abs() < 1e-6);
        }
    
        #[test]
        fn many_same_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
    
            for _ in 0..10_000 {
                vqf.update(gyr, acc, None);
            }
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.887649).abs() < 1e-6); //Look into why there is so
            assert!((quat.1 - 0.334951).abs() < 1e-6); // much difference between them
            assert!((quat.2 - -0.314853).abs() < 1e-6); // we use f32 math, they use mostly double math
            assert!((quat.3 - 0.0274545).abs() < 1e-6);
        }
    
        #[test]
        fn many_different_6D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [4.5, 6.7, 3.2];
    
            for _ in 0..10_000 {
                vqf.update(gyr, acc, None);
            }
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.826852).abs() < 1e-6); //Look into why there is so
            assert!((quat.1 - 0.475521).abs() < 1e-6); // much difference between them
            assert!((quat.2 - -0.299322).abs() < 1e-6); // we use f32 math, they use mostly double math
            assert!((quat.3 - 0.0245133).abs() < 1e-6);
        }
    
        #[test]
        fn single_same_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [10.0, 10.0, 10.0];
            vqf.update_gyr(gyr);
            vqf.update_acc(acc);
            vqf.update_mag(mag);
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.86428).abs() < 1e-6);
            assert!((quat.1 - 0.391089).abs() < 1e-6);
            assert!((quat.2 - -0.241608).abs() < 1e-6);
            assert!((quat.3 - 0.204195).abs() < 1e-6);
        }
    
        #[test]
        fn single_x_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [10.0, 0.0, 0.0];
            vqf.update(gyr, acc, Some(mag));
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.540625).abs() < 1e-6);
            assert!((quat.1 - 0.455768).abs() < 1e-6);
            assert!((quat.2 - 0.060003).abs() < 1e-6);
            assert!((quat.3 - 0.704556).abs() < 1e-6);
        }
    
        #[test]
        fn single_y_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [0.0, 10.0, 0.0];
            vqf.update(gyr, acc, Some(mag));
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.880476).abs() < 1e-6);
            assert!((quat.1 - 0.279848).abs() < 1e-6);
            assert!((quat.2 - -0.364705).abs() < 1e-6);
            assert!((quat.3 - -0.115917).abs() < 1e-6);
        }
    
        #[test]
        fn single_z_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [0.0, 0.0, 10.0];
            vqf.update(gyr, acc, Some(mag));
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.339851).abs() < 1e-6);
            assert!((quat.1 - -0.17592).abs() < 1e-6);
            assert!((quat.2 - -0.424708).abs() < 1e-6);
            assert!((quat.3 - -0.820473).abs() < 1e-6);
        }
    
        #[test]
        fn single_different_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [3.54, 6.32, -2.34];
            vqf.update(gyr, acc, Some(mag));
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.864117).abs() < 1e-6);
            assert!((quat.1 - 0.391281).abs() < 1e-6);
            assert!((quat.2 - -0.241297).abs() < 1e-6);
            assert!((quat.3 - 0.204882).abs() < 1e-6);
        }
    
        #[test]
        fn many_same_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [10.0, 10.0, 10.0];
    
            for _ in 0..10_000 {
                vqf.update(gyr, acc, Some(mag));
            }
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.338005).abs() < 1e-6); //Look into why there is so
            assert!((quat.1 - -0.176875).abs() < 1e-6); // much difference between them
            assert!((quat.2 - -0.424311).abs() < 1e-6); // we use f32 math, they use mostly double math
            assert!((quat.3 - -0.821236).abs() < 1e-6);
        }
    
        #[test]
        fn many_different_9D_quat() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.021, 0.021, 0.021];
            let acc = [5.663806, 5.663806, 5.663806];
            let mag = [3.54, 6.32, -2.34];
    
            for _ in 0..10_000 {
                vqf.update(gyr, acc, Some(mag));
            }
    
            let quat = vqf.quat_9d();
            assert!((quat.0 - 0.864111).abs() < 1e-6); //Look into why there is so
            assert!((quat.1 - 0.391288).abs() < 1e-6); // much difference between them
            assert!((quat.2 - -0.241286).abs() < 1e-6); // we use f32 math, they use mostly double math
            assert!((quat.3 - 0.204906).abs() < 1e-6);
        }
    
        #[test]
        fn run_vqf_cpp_example() {
            let mut vqf = VQF::new(0.01, None, None, None);
    
            let gyr = [0.01745329; 3];
            let acc = [5.663806; 3];
    
            for _ in 0..6000 {
                vqf.update(gyr, acc, None);
            }
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.887781).abs() < 1e-6);
            assert!((quat.1 - 0.333302).abs() < 1e-6);
            assert!((quat.2 - -0.316598).abs() < 1e-6);
            assert!((quat.3 - 0.0228175).abs() < 1e-6);
        }
    
        #[test]
        fn run_vqf_cpp_example_basic() {
            let mut vqf = VQF::new(0.01, None, None, Some(Params {
                rest_bias_est_enabled: false,
                motion_bias_est_enabled: false,
                mag_dist_rejection_enabled: false,
                ..Default::default()
            }));
    
            let gyr = [0.01745329; 3];
            let acc = [5.663806; 3];
    
            for _ in 0..6000 {
                vqf.update(gyr, acc, None);
            }
    
            let quat = vqf.quat_6d();
            assert!((quat.0 - 0.547223).abs() < 1e-6);
            assert!((quat.1 - 0.456312).abs() < 1e-6);
            assert!((quat.2 - 0.055717).abs() < 1e-6);
            assert!((quat.3 - 0.699444).abs() < 1e-6);
        }
    }
}
