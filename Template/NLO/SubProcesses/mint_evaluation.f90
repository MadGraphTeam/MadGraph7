module mint_evaluation
  ! What an integrand evaluation does besides returning its weights.
  ! Ordinary points are integration points; Born-spreading calibration
  ! points and sector-learning probes are side evaluations. The call site
  ! of each side effect checks its field of point_evaluation, which
  ! mint_module sets.
  implicit none
  type :: evaluation_mode
     ! Counted in the grid hits, split over the learned sector cells, and
     ! evaluates the exact one-loop virtual.
     logical :: integration=.true.
     ! Evaluates H events, not only the S weights.
     logical :: h_events=.true.
     ! Fills the MC-integer channel grid.
     logical :: channel_grid=.true.
     ! Counts and reports rejected points.
     logical :: rejection_stats=.true.
     ! In MINT0, evaluates every fold of the point (the probe stencil).
     logical :: all_folds=.false.
     ! Records Born-spreading calibration samples.
     logical :: born_spread_samples=.false.
     ! Records sector-learning probe contributions.
     logical :: learning_probe=.false.
  end type evaluation_mode
  type(evaluation_mode),parameter :: integration_evaluation=evaluation_mode()
  type(evaluation_mode),parameter :: born_spread_evaluation=evaluation_mode( &
       integration=.false.,h_events=.false.,born_spread_samples=.true.)
  type(evaluation_mode),parameter :: learning_probe_evaluation=evaluation_mode( &
       integration=.false.,h_events=.false.,channel_grid=.false.,rejection_stats=.false., &
       all_folds=.true.,learning_probe=.true.)
  type(evaluation_mode) :: point_evaluation=evaluation_mode()
end module mint_evaluation
