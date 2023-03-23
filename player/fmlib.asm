; This is WLA-Z80 syntax, without sections

.struct fmlib_channel
  dataOffset  dw ; Current data offset
  loopOffset  dw ; Offset of loop data, 0 if none seen
  pauseLength dw ; Frame count for current pause
  savedDataOffset dw ; Backup while using a "compressed" run
  compressedDataLength db ; How many bytes to consume in the current "compressed" run
  registers dsb 3 ; $1x, $2x, $3x registers
.endst

.ifndef FMLIB_MEMORY_START
.fail "FMLIB_MEMORY_START not defined"
.endif

.define FMLIB_PORT_REGISTER $f0
.define FMLIB_PORT_DATA $f1
.define FMLIB_PORT_AUDIO_CONTROL $f2
.ifndef FMLIB_MEMORY_CONTROL_ADDRESS
.define FMLIB_MEMORY_CONTROL_ADDRESS $c000 ; default
.endif
.define FMLIB_PORT_MEMORY_CONTROL $3e

.enum FMLIB_MEMORY_START
  fmlib_dataStart dw ; Could reuse? only needed in start
  fmlib_channels instanceof fmlib_channel 10 ; 0-8 tone channels, plus rhythm
.ende

_fmlib_getByte:
  ; Returns next byte in a. Leaves HL pointing at it.
  ; If the next byte is a compression run, deals with it.

;  ; Check for end of compression
;  ld a,(ix+fmlib_channel.compressedDataLength)
;  or a
;  jr z,+
;  ; Non-zero -> decrement
;  dec a
;  ld (ix+fmlib_channel.compressedDataLength),a
;  ; If it reaches zero, we want to restore the old pointer
;  jr nz,+
;  ld l,(ix+fmlib_channel.savedDataOffset+0)
;  ld h,(ix+fmlib_channel.savedDataOffset+1)
;  jr ++
+:
  ; Read pointer
  ld l,(ix+fmlib_channel.dataOffset+0)
  ld h,(ix+fmlib_channel.dataOffset+1)
++:
  ; Get byte
  ld a,(hl)
;  ; Check for start of compression
;  and %11000000
;  cp %11000000
;  jr z,_compressed
_readRawByte:
  ; Read byte
  ld a,(hl)
  ; Bump saved pointer
  inc hl
  ld (ix+fmlib_channel.dataOffset+0),l
  ld (ix+fmlib_channel.dataOffset+1),h
  ; Return hl pointing at the value
  dec hl
  ret

_compressed:
  ; Read length
  ld a,(hl)
  and %00111111
  add 3 ; value is count+4, but we will consume one immediately
  ld (ix+fmlib_channel.compressedDataLength),a
  ; Read pointer
  inc hl
  ld e,(hl)
  inc hl
  ld d,(hl)
  inc hl
  ; Save old pointer
  ld (ix+fmlib_channel.savedDataOffset+0),l
  ld (ix+fmlib_channel.savedDataOffset+1),h
  ; Get new pointer in hl
  ex de,hl
  ; Read from there
  jr _readRawByte
  

fmlib_start:
  ; Args: hl = pointer to data start
  ; Uses: af, bc, de, hl, ix
  
  ld (fmlib_dataStart),hl
  
  ; Reset the channels to 0
  push hl
    ld bc,_sizeof_fmlib_channels - 1
    ld hl,fmlib_channels
    ld de,fmlib_channels+1
    ld (hl),0
    ldir
  pop hl
  
  ; Load the pointers
  ld ix,fmlib_channels
  ld b,10

-:
  ; Read next pointer
  ld e,(hl)
  inc hl
  ld d,(hl)
  inc hl

  ; Make it absolute
  push hl
    ld hl,(fmlib_dataStart)
    add hl,de
    ex de,hl
  pop hl
  ; Save it
  ld (ix+fmlib_channel.dataOffset+1),d
  ld (ix+fmlib_channel.dataOffset+0),e
  
  ; Bump ix
  ld de,_sizeof_fmlib_channel
  add ix,de
  
  ; loop
  djnz -
  
  ; Enable FM
  ld a,%00000011
  out (FMLIB_PORT_AUDIO_CONTROL),a
  
  ret
  
fmlib_play:
  ld a,(FMLIB_MEMORY_CONTROL_ADDRESS)
  or %00000100 ; disable I/O chip so we can access YM2413
  out (FMLIB_PORT_MEMORY_CONTROL),a
  ; Uses: af, bc, de, hl, ix
  ld b,0 ; First tone channel
  ld ix,fmlib_channels
-:
  ; Read channel pointer
  ld l,(ix+fmlib_channel.dataOffset+0)
  ld h,(ix+fmlib_channel.dataOffset+1)
  ; Check for 0 pointer
  ld a,h
  or l
  jr z,_done
  
  ; Check if channel is pausing
  ld e,(ix+fmlib_channel.pauseLength+0)
  ld d,(ix+fmlib_channel.pauseLength+1)
  ld a,d
  or e
  jr z,+
  
  ; We are in a pause
  dec de
  ld (ix+fmlib_channel.pauseLength+0),e
  ld (ix+fmlib_channel.pauseLength+1),d
  jr _done
  
+:; Load some data
  call _fmlib_readData_tone

_done:
  ; Move to next channel
  push ix
  pop hl
  ld de,_sizeof_fmlib_channel
  add hl,de
  push hl
  pop ix
  inc b
  ld a,b
  cp 1;9
  jr nz,-
  ;call _fmlib_readData_noise
 
  ; Re-enable I/O chip
  ld a,(FMLIB_MEMORY_CONTROL_ADDRESS)
  out (FMLIB_PORT_MEMORY_CONTROL),a
  ; Return from fmlib_play
  ret
  
_fmlib_readData_tone:
  call _fmlib_getByte
  and %11000000
  jr nz,_not00
  
  ; %00sxxyyy small tone change + wait
  ld c,a
  ; Get pause
  and %00000111
  ld (ix+fmlib_channel.pauseLength+0),a
  xor a
  ld (ix+fmlib_channel.pauseLength+1),a
  ; Get delta F-num. This is stored as value-1 and sign as we don't need to store 0.
  ; Get value...
  ld a,c
  srl a
  srl a
  srl a
  and %11
  inc a
  ld d,0 ; high bits for 9-bit maths
  ; Get sign, negate as needed
  bit 5,c
  jr z,+
  neg
  dec d ; to $ff
+:; Move to de and extend to 16 bits
  ld e,a
  ; Then add to the registers as a word
  ld l,(ix+fmlib_channel.registers+0)
  ld h,(ix+fmlib_channel.registers+1)
  add hl,de
  ld (ix+fmlib_channel.registers+0),l
  ld (ix+fmlib_channel.registers+1),h
  ; And emit it. We emit both bytes (for now)
  ld a,b
  or $20
  out (FMLIB_PORT_REGISTER),a
  ; YM2413AM says it needs 12 cycles after setting address, 84 cycles after setting data.
  ; out (nn),a is 10 cycles, so we only need 2 more here.
  ld a,h
  out (FMLIB_PORT_DATA),a
  ; But we need 74 here! We should try to move work into this wait. TODO
  ld a,b ; 4
  or $10 ; 7
  call _delay63
  out (FMLIB_PORT_REGISTER),a
  ld a,l
  out (FMLIB_PORT_DATA),a
  ; Assume next will be long enough later?
  ; And done
  inc hl
  ret
  
_not00:
  ld a,(hl)
  and %11100000
  sub %01000000
  jr nz,_not010
  
  ; %010xxxxx wait
  ld a,(hl)
  and %00011111
  jr nz,+
  ; Zero -> read another byte
  call _fmlib_getByte
  ; Add 30 and save
  ld e,a
  ld d,0
  ld hl,30
  add hl,de
  ld (ix+fmlib_channel.pauseLength+0),l
  ld (ix+fmlib_channel.pauseLength+1),h
  ; And done
  ret
  
+:; non-zero -> use as-is
  ld (ix+fmlib_channel.pauseLength+0),a
  xor a
  ld (ix+fmlib_channel.pauseLength+1),a
  ; And done
  ret
  
_not010:
  sub %00100000
  jr nz,_not011
  ; instrument or volume change
  ; check type
  ld a,(hl)
  bit 4,a
  jr z,+
  ; instrument, merge with volume nibble
  ; Shift to high nibble
  .repeat 4
    add a,a
  .endr
  and $f0
  push af
    ; Emit first write while a is free
    ld a,b
    or $30
    out (FMLIB_PORT_REGISTER),a
    ; Capture low nibble in c
    ld a,(ix+fmlib_channel.registers+2)
    and $0f
-:  ld c,a
  pop af
  or c
  ld (ix+fmlib_channel.registers+2),a
  out (FMLIB_PORT_DATA),a
  jp _fmlib_readData_tone
  
+:; volume, merge with instrument nibble
  and $0f
  push af
    ; Emit first write while a is free
    ld a,b
    or $30
    out (FMLIB_PORT_REGISTER),a
    ; Capture high nibble in c
    ld a,(ix+fmlib_channel.registers+2)
    and $f0
    jr - ; code is the same from here
  
_not011:
  ; At this point a is the command byte masked to %11100000, minus %0110000.
  ; Thus if it was originally %100----- then it will now be %00100000
  sub %00100000
  jr nz,_not100
  ; special event
  ld a,(hl)
  and %00011111
  jr nz,+
  ; 0 => key up
  ; Clear the bit
  ld a,(ix+fmlib_channel.registers+1)
  and %11101111
  ld a,(ix+fmlib_channel.registers+1)
  ; Emit it
  push af
    ld a,b
    or $20
    out (FMLIB_PORT_REGISTER),a
  pop af
  out (FMLIB_PORT_DATA),a
  jp _fmlib_readData_tone
  
+:dec a
  jp z,_endOfStream
  ; 1 => end of stream, disable channel
  
  dec a
  jr nz,+
  ; 2 => custom instrument data
  ld d,0
-:ld a,d
  out (FMLIB_PORT_REGISTER),a
  call _fmlib_getByte
  out (FMLIB_PORT_DATA),a
  inc d
  ; If it gets to 8, we are done
  ld a,d
  cp 8
  jr nz,-
  jp _fmlib_readData_tone
  
+:; 3 => loop marker (we assume it is 3)
  inc hl
  ld (ix+fmlib_channel.loopOffset+0),l
  ld (ix+fmlib_channel.loopOffset+1),h
  jp _fmlib_readData_tone  

_not100:
  ; Since compressed bytes are handled elsewhere, we can assume 
  ; this is %101sbbbf ffffffff = full note + key down
  ld a,$20
  or b
  out (FMLIB_PORT_REGISTER),a
  ; We need to move the sustain bit one to the left...
  ld a,(hl)
  and %00010000
  add a,a
  ; then mix it back in
  ld c,a
  ld a,(hl)
  and %00001111
  or c
  ; and set the key bit
  or %00010000
  out (FMLIB_PORT_DATA),a
  ld (ix+fmlib_channel.registers+1),a
  ; Then the next byte goes to $1x
  ld a,$10
  or b ; Is it cheaper to ld a,b; set 4,a? TODO
  out (FMLIB_PORT_REGISTER),a
  call _fmlib_getByte
  ld (ix+fmlib_channel.registers+0),a
  out (FMLIB_PORT_DATA),a
  ; And done
  jp _fmlib_readData_tone
  

_fmlib_readData_noise:
  ; It's the noise channel
  call _fmlib_getByte
  bit 7,a
  jr nz,_not0x
  ; %0nnkkkkk keys + wait
  ld a,$0e
  out (FMLIB_PORT_REGISTER),a
  ld a,(hl)
  ; Enable rhythm mode
  and %00011111
  or  %00100000
  ; Emit
  out (FMLIB_PORT_DATA),a
  ; Get pause
  sla a
  sla a
  sla a
  and %00000011
_savePauseLength:
  ; TODO reuse this elsewhere
  ld (ix+fmlib_channel.pauseLength+0),a
  xor a
  ld (ix+fmlib_channel.pauseLength+1),a
  ; And done
  ret

_not0x:
  and %01100000
  jr nz,_not100r
  ; %100kkkkk
  ld a,$0e
  out (FMLIB_PORT_REGISTER),a
  ld a,(hl)
  ; Enable rhythm mode
  and %00011111
  or  %00100000
  ; Emit
  out (FMLIB_PORT_DATA),a
  ; Move to next byte
  jp _fmlib_readData_noise
  
_not100r:
  ; Must be %101xxxxx
  ld a,(hl)
  and %00011111
  jr nz,_savePauseLength ; and done
  ; Zero pause is a control byte
  call _fmlib_getByte
  or a
  ; zero = end of stream
  jp z,_endOfStream
  ; Else it's the loop point
  inc hl
  ld (ix+fmlib_channel.loopOffset+0),l
  ld (ix+fmlib_channel.loopOffset+1),h
  jp _fmlib_readData_noise

_endOfStream:
  ; Disable channel
  xor a
  ld (ix+fmlib_channel.dataOffset+0),a
  ld (ix+fmlib_channel.dataOffset+1),a
  ; And done
  ret

_delay63:
  ; call and ret cost 27, we need to waste 36
  .repeat 9
  nop
  .endr
  ret
  