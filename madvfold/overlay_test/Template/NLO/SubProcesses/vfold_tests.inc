      ! MadVfold: development and test hooks (e.g. the FVZ dumps) are
      ! enabled in 'output --vfold-with-tests' (this file replaces the copy
      ! in madvfold/overlay, installed by --vfold, which disables them)
      logical vfold_tests
      parameter (vfold_tests=.true.)
