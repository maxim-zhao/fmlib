; WLA-DX banking setup
.memorymap
defaultslot 0
slotsize 16*1024 ; ROM
slot 0 $0000
slot 1 $4000
slot 2 $8000
slotsize 8*1024 ; RAM
slot 3 $c000
.endme

; You must specify your ROM size here. Mostly you only need two, but I put more here to demonstrate using paging.
.rombankmap
bankstotal 2
banksize 16*1024
banks 2
.endro

.define VDPData $be
.define VDPAddress $bf
.define VDPRegister $bf

; VDP register names
.enum $80
VDPReg_0                 db ; Misc
VDPReg_1                 db ; Misc
VDPRegTileMapAddress     db
VDPReg_3                 db ; Unused
VDPReg_4                 db ; Unused
VDPRegSpriteTableAddress db
VDPRegSpriteTileSet      db
VDPRegBorderColour       db
VDPRegHScroll            db
VDPRegVScroll            db
VDPRegLineInt            db
.ende


; We set some definitions for the way we're using VRAM.
.define SpriteSet 1 ; Use upper 256 tiles for sprites
; These are the standard locations and it's best not to use anything else unless you really need to.
.define TileMapAddress     $3800
.define SpriteTableAddress $3f00

; Here is where we specify our RAM usage. This also allows the debugging emulator to show names instead of addresses.
.enum $c000 export
.ende

.bank 0 slot 0

; This sets the necessar metadata for the game to work on a real system, and also to let you get credit for your work.
.sdsctag 0.01, "fmlib demo", "", "Maxim"

.org 0
; standard startup
.section "Startup" force
  di
  im 1
  ld sp, $dff0
  jp Initialise
.ends

.org $38
.section "Interrupt handler" force
InterruptHandler:
  ex af,af'
  exx
    in a,($bf)
    and %10000000
    call nz, fmlib_play
  exx
  ex af,af'
  ei
  reti
.ends

.org $66
.section "Pause handler" force
PauseHandler:
  retn
.ends

.section "Initialisation" free
Initialise:
  ; Initialise the VDP
  call InitialiseVDPRegisters
  call ClearVRAM

  ; Clear RAM. We leave the first byte alone, as that is the memory control value, and the last 16 bytes, as we don't use them.
  ld hl, $c001 ; First byte to clear
  ld de, $c002 ; Second byte to clear
  ld bc, $dff0 - $c002 ; Count
  ld (hl), 0; Set first byte to 0
  ldir ; ...and copy that forward
  
  ; Initialise paging
  xor a
  ld ($fffc),a
  ld ($fffd),a
  inc a
  ld ($fffe),a
  inc a
  ld ($ffff),a

  ; Load palette
  ld hl,Palette
  xor a
  out (VDPAddress),a
  ld a,$c0
  out (VDPAddress),a
  ld b,32
  ld c,VDPData
  otir
  
  call TurnOnScreen

  ld hl,Data
  call fmlib_start
  
  ei

-:halt
  jr -

Palette:
.db $00, $3f
.dsb 30 $00
.ends

.section "Screen control" free
; Design decision: I'm never going to vary the features controlled by this register.
.define VDP_REG_1_VALUE_SCREEEN_OFF %10100000
.define VDP_REG_1_VALUE_SCREEEN_ON  %11100000
                                ;    |||||||`- Zoomed sprites -> 16x16 pixels
                                ;    ||||||`-- Doubled sprites -> 2 tiles per sprite, 8x16
                                ;    |||||`--- Always 0
                                ;    ||||`---- 30 row/240 line mode
                                ;    |||`----- 28 row/224 line mode
                                ;    ||`------ Enable VBlank interrupts
                                ;    |`------- Enable display
                                ;    `-------- Always 1
TurnOffScreen:
  ld a, VDP_REG_1_VALUE_SCREEEN_OFF
  jr +
TurnOnScreen:
  ld a, VDP_REG_1_VALUE_SCREEEN_ON
+:out (VDPRegister), a
  ld a, VDPReg_1
  out (VDPRegister), a
  ret
.ends

.section "Initialise VDP registers" free
; Set all the VDP registers. We start with the screen turned off and VDP interrupts disabled.
; No arguments.
; Alters hl, bc
InitialiseVDPRegisters:
  ld hl,_Data
  ld b,_End-_Data
  ld c,VDPRegister
  otir
  ret
  
; Let's do some sanity checks on the definitions...
.if SpriteSet >> 1 != 0
.fail "SpriteSet must be either 0 or 1"
.endif
.if (TileMapAddress & %100011111111111) != 0
.fail "TileMapAddress must be a multiple of $800 between 0 and $3800 (usually $3800)"
.endif
.if SpriteTableAddress & %1100000011111111 != 0
.fail "SpriteTableAddress must be a multiple of $100 between 0 and $3f00 (usually $3f00)"
.endif
  
_Data:
  .db %00100110, VDPReg_0
  ;    |||||||`- Disable sync
  ;    ||||||`-- Enable extra height modes
  ;    |||||`--- SMS mode instead of SG
  ;    ||||`---- Shift sprites left 8 pixels
  ;    |||`----- Enable line interrupts
  ;    ||`------ Blank leftmost column for scrolling
  ;    |`------- Fix top 2 rows during horizontal scrolling
  ;    `-------- Fix right 8 columns during vertical scrolling
  .db VDP_REG_1_VALUE_SCREEEN_OFF, VDPReg_1 ; See above
  .db (TileMapAddress >> 10)    | %11110001, VDPRegTileMapAddress
  .db (SpriteTableAddress >> 7) | %10000001, VDPRegSpriteTableAddress
  .db (SpriteSet << 2)          | %11111011, VDPRegSpriteTileSet
  .db $0 | $f0, VDPRegBorderColour
  ;    `-------- Border palette colour (sprite palette)
  .db $00, VDPRegHScroll
  ;    ``------- Horizontal scroll
  .db $00, VDPRegVScroll
  ;    ``------- Vertical scroll
  .db $ff, VDPRegLineInt
  ;    ``------- Line interrupt spacing ($ff to disable)
_End:
.ends

.section "Clear VRAM" free
; Writes zero to all of VRAM.
; No arguments.
; Alters bc, af
ClearVRAM:
  ld a,0
  out (VDPAddress),a
  ld a,$40
  out (VDPAddress),a
  ; Output 16KB of zeroes
  ld bc,16*1024    ; Counter
-:xor a            ; Value to write
  out (VDPData),a  ; Output it
  dec bc           ; Decrement counter
  ld a,b           ; Loop until it is zero
  or c
  jr nz,-
  ret
.ends

.section "Data" superfree
Data:
.incbin "data.vgm.fm"
/*
.dw _ch0-Data, 0, 0, 0, 0, 0, 0, 0, 0, 0
_ch0:
.db %01100000 ; volume to 0 => max
.db %01110111 ; instrument 7
.db %10101001 %00000001 ; a tone with key down
.repeat 100
.db %01000000 %11111111 ; a long wait
.endr
.db %10000001 ; end of stream
*/
.ends

.section "fmlib" free
.define FMLIB_MEMORY_START $c010
.include "fmlib.asm"
.ends