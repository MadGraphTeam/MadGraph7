      ! MadVFold: development and test hooks (e.g. the FVZ dumps) are
      ! disabled in 'output --vfold' (they are enabled by the copy of this
      ! file in madvfold/overlay_test, installed by --vfold-with-tests)
      logical vfold_tests
      parameter (vfold_tests=.false.)
