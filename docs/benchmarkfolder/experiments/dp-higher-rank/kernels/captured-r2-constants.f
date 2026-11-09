      MODULE TEST_POLYNOMIAL_CONSTANTS
      IMPLICIT NONE
      INCLUDE 'coef_specs.inc'
      INCLUDE 'loop_max_coefs.inc'

C     Map associating a rank to each coefficient position
      INTEGER COEFTORANK_MAP(0:LOOPMAXCOEFS-1)
      DATA COEFTORANK_MAP(0:0)/1*0/
      DATA COEFTORANK_MAP(1:4)/4*1/
      DATA COEFTORANK_MAP(5:14)/10*2/
      DATA COEFTORANK_MAP(15:34)/20*3/
      DATA COEFTORANK_MAP(35:69)/35*4/

C     Map defining the number of coefficients for a symmetric tensor
C      of a given rank
      INTEGER NCOEF_R(0:4)
      DATA NCOEF_R/1,5,15,35,70/

C     Map defining the coef position resulting from the multiplication
C      of two lower rank coefs.
      INTEGER COMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)
      DATA COMB_COEF_POS(  0,  0:  4) /  0,  1,  2,  3,  4/
      DATA COMB_COEF_POS(  1,  0:  4) /  1,  5,  6,  8, 11/
      DATA COMB_COEF_POS(  2,  0:  4) /  2,  6,  7,  9, 12/
      DATA COMB_COEF_POS(  3,  0:  4) /  3,  8,  9, 10, 13/
      DATA COMB_COEF_POS(  4,  0:  4) /  4, 11, 12, 13, 14/
      DATA COMB_COEF_POS(  5,  0:  4) /  5, 15, 16, 19, 25/
      DATA COMB_COEF_POS(  6,  0:  4) /  6, 16, 17, 20, 26/
      DATA COMB_COEF_POS(  7,  0:  4) /  7, 17, 18, 21, 27/
      DATA COMB_COEF_POS(  8,  0:  4) /  8, 19, 20, 22, 28/
      DATA COMB_COEF_POS(  9,  0:  4) /  9, 20, 21, 23, 29/
      DATA COMB_COEF_POS( 10,  0:  4) / 10, 22, 23, 24, 30/
      DATA COMB_COEF_POS( 11,  0:  4) / 11, 25, 26, 28, 31/
      DATA COMB_COEF_POS( 12,  0:  4) / 12, 26, 27, 29, 32/
      DATA COMB_COEF_POS( 13,  0:  4) / 13, 28, 29, 30, 33/
      DATA COMB_COEF_POS( 14,  0:  4) / 14, 31, 32, 33, 34/
      DATA COMB_COEF_POS( 15,  0:  4) / 15, 35, 36, 40, 50/
      DATA COMB_COEF_POS( 16,  0:  4) / 16, 36, 37, 41, 51/
      DATA COMB_COEF_POS( 17,  0:  4) / 17, 37, 38, 42, 52/
      DATA COMB_COEF_POS( 18,  0:  4) / 18, 38, 39, 43, 53/
      DATA COMB_COEF_POS( 19,  0:  4) / 19, 40, 41, 44, 54/
      DATA COMB_COEF_POS( 20,  0:  4) / 20, 41, 42, 45, 55/
      DATA COMB_COEF_POS( 21,  0:  4) / 21, 42, 43, 46, 56/
      DATA COMB_COEF_POS( 22,  0:  4) / 22, 44, 45, 47, 57/
      DATA COMB_COEF_POS( 23,  0:  4) / 23, 45, 46, 48, 58/
      DATA COMB_COEF_POS( 24,  0:  4) / 24, 47, 48, 49, 59/
      DATA COMB_COEF_POS( 25,  0:  4) / 25, 50, 51, 54, 60/
      DATA COMB_COEF_POS( 26,  0:  4) / 26, 51, 52, 55, 61/
      DATA COMB_COEF_POS( 27,  0:  4) / 27, 52, 53, 56, 62/
      DATA COMB_COEF_POS( 28,  0:  4) / 28, 54, 55, 57, 63/
      DATA COMB_COEF_POS( 29,  0:  4) / 29, 55, 56, 58, 64/
      DATA COMB_COEF_POS( 30,  0:  4) / 30, 57, 58, 59, 65/
      DATA COMB_COEF_POS( 31,  0:  4) / 31, 60, 61, 63, 66/
      DATA COMB_COEF_POS( 32,  0:  4) / 32, 61, 62, 64, 67/
      DATA COMB_COEF_POS( 33,  0:  4) / 33, 63, 64, 65, 68/
      DATA COMB_COEF_POS( 34,  0:  4) / 34, 66, 67, 68, 69/
      DATA COMB_COEF_POS( 35,  0:  4) / 35, 70, 71, 76, 91/
      DATA COMB_COEF_POS( 36,  0:  4) / 36, 71, 72, 77, 92/
      DATA COMB_COEF_POS( 37,  0:  4) / 37, 72, 73, 78, 93/
      DATA COMB_COEF_POS( 38,  0:  4) / 38, 73, 74, 79, 94/
      DATA COMB_COEF_POS( 39,  0:  4) / 39, 74, 75, 80, 95/
      DATA COMB_COEF_POS( 40,  0:  4) / 40, 76, 77, 81, 96/
      DATA COMB_COEF_POS( 41,  0:  4) / 41, 77, 78, 82, 97/
      DATA COMB_COEF_POS( 42,  0:  4) / 42, 78, 79, 83, 98/
      DATA COMB_COEF_POS( 43,  0:  4) / 43, 79, 80, 84, 99/
      DATA COMB_COEF_POS( 44,  0:  4) / 44, 81, 82, 85,100/
      DATA COMB_COEF_POS( 45,  0:  4) / 45, 82, 83, 86,101/
      DATA COMB_COEF_POS( 46,  0:  4) / 46, 83, 84, 87,102/
      DATA COMB_COEF_POS( 47,  0:  4) / 47, 85, 86, 88,103/
      DATA COMB_COEF_POS( 48,  0:  4) / 48, 86, 87, 89,104/
      DATA COMB_COEF_POS( 49,  0:  4) / 49, 88, 89, 90,105/
      DATA COMB_COEF_POS( 50,  0:  4) / 50, 91, 92, 96,106/
      DATA COMB_COEF_POS( 51,  0:  4) / 51, 92, 93, 97,107/
      DATA COMB_COEF_POS( 52,  0:  4) / 52, 93, 94, 98,108/
      DATA COMB_COEF_POS( 53,  0:  4) / 53, 94, 95, 99,109/
      DATA COMB_COEF_POS( 54,  0:  4) / 54, 96, 97,100,110/
      DATA COMB_COEF_POS( 55,  0:  4) / 55, 97, 98,101,111/
      DATA COMB_COEF_POS( 56,  0:  4) / 56, 98, 99,102,112/
      DATA COMB_COEF_POS( 57,  0:  4) / 57,100,101,103,113/
      DATA COMB_COEF_POS( 58,  0:  4) / 58,101,102,104,114/
      DATA COMB_COEF_POS( 59,  0:  4) / 59,103,104,105,115/
      DATA COMB_COEF_POS( 60,  0:  4) / 60,106,107,110,116/
      DATA COMB_COEF_POS( 61,  0:  4) / 61,107,108,111,117/
      DATA COMB_COEF_POS( 62,  0:  4) / 62,108,109,112,118/
      DATA COMB_COEF_POS( 63,  0:  4) / 63,110,111,113,119/
      DATA COMB_COEF_POS( 64,  0:  4) / 64,111,112,114,120/
      DATA COMB_COEF_POS( 65,  0:  4) / 65,113,114,115,121/
      DATA COMB_COEF_POS( 66,  0:  4) / 66,116,117,119,122/
      DATA COMB_COEF_POS( 67,  0:  4) / 67,117,118,120,123/
      DATA COMB_COEF_POS( 68,  0:  4) / 68,119,120,121,124/
      DATA COMB_COEF_POS( 69,  0:  4) / 69,122,123,124,125/

      END MODULE TEST_POLYNOMIAL_CONSTANTS

