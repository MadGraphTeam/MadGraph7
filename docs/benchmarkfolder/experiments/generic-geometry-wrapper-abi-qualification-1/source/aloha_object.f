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
