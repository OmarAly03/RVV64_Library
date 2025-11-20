.section .data,"aw",@progbits
.global SIZE; SIZE: .word 10
.global input_data
.section .data
.balign 256
input_data:
    .word 0x3ebfc3b9
    .word 0x3f736203
    .word 0x3f3b63f4
    .word 0x3f1941af
    .word 0x3e1fc35a
    .word 0x3e1fbd07
    .word 0x3d6de915
    .word 0x3f5dbdb8
    .word 0x3f19e2ac
    .word 0x3f35443f
.global golden_data
.section .data
.balign 256
golden_data:
    .word 0x3da9843c
    .word 0x3e16cdb0
    .word 0x3df25aeb
    .word 0x3dd41a32
    .word 0x3d883dd3
    .word 0x3d883cfb
    .word 0x3d770ffd
    .word 0x3e0a942e
    .word 0x3dd49fbf
    .word 0x3deca062
.global output_data
.section .data
.balign 256
output_data:
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
    .word 0x00000000
