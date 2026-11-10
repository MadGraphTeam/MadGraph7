C###############################################################################
C
C Copyright (c) 2010 The ALOHA Development team and Contributors
C
C This file is a part of the MadGraph7 project, an application which
C automatically generates Feynman diagrams and matrix elements for arbitrary
C high-energy processes in the Standard Model and beyond.
C
C It is subject to the ALOHA license which should accompany this
C distribution.
C
C###############################################################################
      module ALOHA_OBJECT
         TYPE ALOHA
            SEQUENCE
            double complex::W(4)
            double complex :: P(0:3)
            integer :: flv_index
         END TYPE ALOHA
         TYPE ALOHA2D
            SEQUENCE
            double complex::W(16)
            double complex :: P(0:3)
            integer :: flv_index
         END TYPE ALOHA2D
         TYPE MP_ALOHA
            SEQUENCE
            complex*32 :: W(4)
            complex*32 :: P(0:3)
            integer :: flv_index
         END TYPE MP_ALOHA
         TYPE MP_ALOHA2D
            SEQUENCE
            complex*32 :: W(16)
            complex*32 :: P(0:3)
            integer :: flv_index
         END TYPE MP_ALOHA2D
      end module ALOHA_OBJECT

