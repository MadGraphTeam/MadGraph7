module avh_olo_forCutTools_dp_tri
  implicit none
  integer ,parameter :: permtable(3,0:7)=reshape((/ &
       1,2,3 &! 0, 0 masses non-zero, no permutation
      ,1,2,3 &! 1, 1 mass non-zero,   no permutation
      ,3,1,2 &! 2, 1 mass non-zero,   1 cyclic permutation
      ,1,2,3 &! 3, 2 masses non-zero, no permutation
      ,2,3,1 &! 4, 1 mass non-zero,   2 cyclic permutations
      ,2,3,1 &! 5, 2 masses non-zero, 2 cyclic permutations
      ,3,1,2 &! 6, 2 masses non-zero, 1 cyclic permutation
      ,1,2,3 &! 7, 3 masses non-zero, no permutation
      /) ,(/3,8/))                     ! 0,1,2,3,4,5,6,7
  integer ,parameter :: casetable(0:7)=(/0,1,1,2,1,2,2,3/)
  integer ,parameter :: base(3)=(/4,2,1/)

end module

module avh_olo_forCutTools_dp_box
  implicit none
  integer ,parameter ::  permtable(6,0:15)=reshape((/ &
     1,2,3,4 ,5,6 &! 0, 0 masses non-zero,           no perm
    ,1,2,3,4 ,5,6 &! 1, 1 mass non-zero,             no perm
    ,4,1,2,3 ,6,5 &! 2, 1 mass non-zero,             1 cyclic perm
    ,1,2,3,4 ,5,6 &! 3, 2 neighbour masses non-zero, no perm
    ,3,4,1,2 ,5,6 &! 4, 1 mass   non-zero,           2 cyclic perm's
    ,1,2,3,4 ,5,6 &! 5, 2 opposite masses non-zero,  no perm
    ,4,1,2,3 ,6,5 &! 6, 2 neighbour masses non-zero, 1 cyclic perm
    ,1,2,3,4 ,5,6 &! 7, 3 masses non-zero,           no perm
    ,2,3,4,1 ,6,5 &! 8, 1 mass   non-zero,           3 cyclic perm's
    ,2,3,4,1 ,6,5 &! 9, 2 neighbour masses non-zero, 3 cyclic perm's
    ,4,1,2,3 ,6,5 &!10, 2 opposite masses non-zero,  1 cyclic perm
    ,2,3,4,1 ,6,5 &!11, 3 masses non-zero,           3 cyclic perm's
    ,3,4,1,2 ,5,6 &!12, 2 neighbour masses non-zero, 2 cyclic perm's
    ,3,4,1,2 ,5,6 &!13, 3 masses non-zero,           2 cyclic perm's
    ,4,1,2,3 ,6,5 &!14, 3 masses non-zero,           1 cyclic perm
    ,1,2,3,4 ,5,6 &!15, 4 masses non-zero,           no perm
    /),(/6,16/)) !          0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  integer ,parameter :: casetable(0:15)= &
                          (/0,1,1,2,1,5,2,3,1,2, 5, 3, 2, 3, 3, 4/)
  integer ,parameter :: base(4)=(/8,4,2,1/)
end module

module avh_olo_forCutTools_qp_tri
  implicit none
  integer ,parameter :: permtable(3,0:7)=reshape((/ &
       1,2,3 &! 0, 0 masses non-zero, no permutation
      ,1,2,3 &! 1, 1 mass non-zero,   no permutation
      ,3,1,2 &! 2, 1 mass non-zero,   1 cyclic permutation
      ,1,2,3 &! 3, 2 masses non-zero, no permutation
      ,2,3,1 &! 4, 1 mass non-zero,   2 cyclic permutations
      ,2,3,1 &! 5, 2 masses non-zero, 2 cyclic permutations
      ,3,1,2 &! 6, 2 masses non-zero, 1 cyclic permutation
      ,1,2,3 &! 7, 3 masses non-zero, no permutation
      /) ,(/3,8/))                     ! 0,1,2,3,4,5,6,7
  integer ,parameter :: casetable(0:7)=(/0,1,1,2,1,2,2,3/)
  integer ,parameter :: base(3)=(/4,2,1/)

end module

module avh_olo_forCutTools_qp_box
  implicit none
  integer ,parameter ::  permtable(6,0:15)=reshape((/ &
     1,2,3,4 ,5,6 &! 0, 0 masses non-zero,           no perm
    ,1,2,3,4 ,5,6 &! 1, 1 mass non-zero,             no perm
    ,4,1,2,3 ,6,5 &! 2, 1 mass non-zero,             1 cyclic perm
    ,1,2,3,4 ,5,6 &! 3, 2 neighbour masses non-zero, no perm
    ,3,4,1,2 ,5,6 &! 4, 1 mass   non-zero,           2 cyclic perm's
    ,1,2,3,4 ,5,6 &! 5, 2 opposite masses non-zero,  no perm
    ,4,1,2,3 ,6,5 &! 6, 2 neighbour masses non-zero, 1 cyclic perm
    ,1,2,3,4 ,5,6 &! 7, 3 masses non-zero,           no perm
    ,2,3,4,1 ,6,5 &! 8, 1 mass   non-zero,           3 cyclic perm's
    ,2,3,4,1 ,6,5 &! 9, 2 neighbour masses non-zero, 3 cyclic perm's
    ,4,1,2,3 ,6,5 &!10, 2 opposite masses non-zero,  1 cyclic perm
    ,2,3,4,1 ,6,5 &!11, 3 masses non-zero,           3 cyclic perm's
    ,3,4,1,2 ,5,6 &!12, 2 neighbour masses non-zero, 2 cyclic perm's
    ,3,4,1,2 ,5,6 &!13, 3 masses non-zero,           2 cyclic perm's
    ,4,1,2,3 ,6,5 &!14, 3 masses non-zero,           1 cyclic perm
    ,1,2,3,4 ,5,6 &!15, 4 masses non-zero,           no perm
    /),(/6,16/)) !          0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  integer ,parameter :: casetable(0:15)= &
                          (/0,1,1,2,1,5,2,3,1,2, 5, 3, 2, 3, 3, 4/)
  integer ,parameter :: base(4)=(/8,4,2,1/)
end module
