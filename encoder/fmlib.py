import gzip;
import copy;
import sys;

class Wait:
    def __init__(self, length):
        self.length = length;

    def __str__(self):
        return f"Wait({self.length})"


class Command:
    def __init__(self, register, data):
        self.register = register;
        self.data = data;

    def __str__(self):
        return f"Command({self.register:#04x}, {self.data:#04x})"

class LoopPoint:
    def __init__(self):
        pass
        
    def __str__(self):
        return "LoopPoint"

class VgmFile:
    def __init__(self, filename):
        with gzip.open(filename, "rb") as f:
            self.data = f.read();

        # Check VGM header
        if self.data[0:4].decode("ASCII") != "Vgm ":
            raise Exception("Not a valid VGM file")

        # Read some of the header
        # We could use the struct library if this becomes more complex
        self.eof_offset = self.read32_relative(0x4)
        self.version = self.read32(0x8) # Leave it as 32-bit rather than convert BCD...
        self.loop_offset = self.read32_relative(0x1c)
        if self.version >= 0x150:
            self.data_offset = self.read32_relative(0x34)
        else:
            self.data_offset = 0x40;

        self.position = self.data_offset


    def read32(self, offset):
        return int.from_bytes(self.data[offset:offset+4], byteorder='little')


    def read32_relative(self, offset):
        value = self.read32(offset)
        return value + offset if value > 0 else 0


    def commands(self):
        while self.position < self.eof_offset:
            if self.position == self.loop_offset:
                yield LoopPoint()
            b = self.data[self.position]
            self.position += 1
            if b == 0x4f:
                pass # GG stereo
            elif b == 0x50:
                self.position += 1
                pass # PSG
            elif b == 0x51: # YM2413
                yield Command(self.data[self.position], self.data[self.position + 1])
                self.position += 2
                pass
            elif b == 0x61: # Wait nnnn
                yield Wait(int.from_bytes(self.data[self.position : self.position + 2], byteorder='little'))
                self.position += 2
                pass
            elif b == 0x62: # Wait 1/60
                yield Wait(735)
                pass
            elif b == 0x62: # Wait 1/50
                yield Wait(882)
                pass
            elif b == 0x66: # End
                return


class Channel:
    # This is a base for ToneChannel and RhythmChannel. It accumulates wait commands.
    def __init__(self):
        self.waits = 0
        self.data = bytearray()

    def wait(self, length):
        # We may get multiple pauses, so we accumulate them
        self.waits += length

    def maybe_write_data(self):
        # We write if we have accumulated pauses
        if self.waits < 735:
            return
        self.write_data()

class ToneChannel(Channel):    
    class ToneState:
        def __init__(self):
            self.fnum = 0
            self.block = 0
            self.key = False
            self.sustain = False
            self.volume = 0
            self.instrument = 0

    def __init__(self, index):
        super().__init__()
        self.channel = index
        self.state = self.ToneState()
        self.last_state = self.ToneState()
        self.custom_instrument = [0, 0, 0, 0, 0, 0, 0, 0]
        self.custom_instrument_changed = False
    
    # When registers are written, we extract the data from them into self.state.
    # We want to dump out the accumulated change only when we get to a wait.
    # However we also want to accumulate that wait together with the change,
    # in the case where the wait is small.
    # Thus, we write to our internal data stream when we see a register write
    # at a time after at least one frame of pauses has been seen; so we call
    # maybe_write_data() before handling each update. This will emit (data)(pause)
    # if pause is >0 frames, and do nothing (i.e. accumulate more data) if it
    # is <1 frame.

    def reg1x(self, b):
        self.maybe_write_data()
        # F-Num low bits
        self.state.fnum = (self.state.fnum & 0x100) | b

    def reg2x(self, b):
        self.maybe_write_data()
        # F-Num high bit
        self.state.fnum = (self.state.fnum & 0x0ff) | ((b & 0b1) << 8)
        # Block
        self.state.block = (b >> 1) & 0b111
        # Key
        self.state.key = b & 0b00010000 != 0
        # Sustain
        self.state.sustain = b & 0b00100000 != 0

    def reg3x(self, b):
        self.maybe_write_data()
        # Volume
        self.state.volume = b & 0b1111
        # Instrument
        self.state.instrument = b >> 4

    def add_custom(self, index, data):
        if self.custom_instrument[index] != data:
            self.custom_instrument[index] = data
            self.custom_instrument_changed = True
            
    def add_loop(self):
        self.write_data()
        self.data.append(0b10000011)
        
    def write_data(self):
        # We want to encode some data for what's changed since last time, then the pause (if non-zero)
        frames = self.waits // 735
        delta = self.state.fnum - self.last_state.fnum

        # If key has gone from down to up, we handle that first
        if (not self.state.key) and self.last_state.key:
            print(f"ch{self.channel}: Key up")
            self.data.append(0b10000000)
            self.last_state.key = self.state.key
            # And we ignore any other changes... this is not quite VGM compatible as we are delaying those changes to the next key down
            delta = 0
        else:
            # Check for a large f-num change or other change affecting block, key(, sustain)
            if delta < -4 or delta > 4 or self.state.block != self.last_state.block or self.state.sustain != self.last_state.sustain or self.state.key != self.last_state.key or frames == 0:
                print(f"ch{self.channel}: Changed from key {self.last_state.key}, sustain {self.last_state.sustain}, block {self.last_state.block}, f-num {self.last_state.fnum} to key {self.state.key}, sustain {self.state.sustain}, block {self.state.block}, f-num {self.state.fnum}")
                b = 0b10100000
                if self.state.sustain:
                    b |= 0b00010000
                b |= self.state.fnum >> 8
                b |= self.state.block << 1
                self.data.append(b)
                self.data.append(self.state.fnum & 0xff)
                # Remember the changes
                self.last_state.key = self.state.key
                self.last_state.sustain = self.state.sustain
                self.last_state.block = self.state.block
                self.last_state.fnum = self.state.fnum
                # This captures 3 or 6 bytes of VGM to 2 bytes
            
            if self.state.volume != self.last_state.volume:
                self.data.append(0b01100000 | self.state.volume)
                self.last_state.volume = self.state.volume
                print(f"ch{self.channel}: volume changed to {self.state.volume}")
            if self.state.instrument != self.last_state.instrument:
                self.data.append(0b01110000 | self.state.instrument)
                self.last_state.instrument = self.state.instrument
                print(f"ch{self.channel}: instrument changed to {self.state.instrument}")
            # These two will expand 3 bytes to 1 or 2 bytes of data
            
        if self.state.fnum != self.last_state.fnum:
            # Must be a small change left over from above, not accompanied by anything else important
            print(f"ch{self.channel}: Changed f-num from {self.last_state.fnum} to {self.state.fnum} ({delta}) and wait is {frames} frames")
            b = 0
            frames_to_consume = min(frames, 8) # We may not consume them all, and emit another byte for the remaining pause
            if delta < 0:
                b |= 0b00100000
                delta *= -1
            b |= delta << 2
            b |= frames_to_consume - 1
            self.data.append(b)
            self.waits -= frames_to_consume * 735
            # This captures 3-4 bytes of VGM to 1 byte
            
        if self.custom_instrument_changed:
            print("Custom instrument changed")
            self.data.append(0b10000010)
            for b in self.custom_instrument:
                self.data.append(b)
            self.custom_instrument_changed = False

        if self.waits >= 735:
            frames = self.waits // 735
            print(f"ch{self.channel}: Pause {frames} frames")
            self.waits -= frames * 735
            while frames > 0:
                b = 0b01000000
                if frames <= 31:
                    # 1-byte count
                    b |= frames
                    self.data.append(b)
                    frames = 0
                elif frames <= 255 + 31:
                    # 2-byte count
                    self.data.append(b)
                    self.data.append(frames - 32)
                    frames = 0
                else:
                    # 2-byte max count and loop
                    self.data.append(b)
                    self.data.append(255)
                    frames -= 255 + 32

    def terminate(self):
        # We have space in the %100----- area
        self.data.append(0b10000001)
    

class RhythmChannel(Channel):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.value_changed = False
        
    def write(self, data):
        self.maybe_write_data()
        self.value = data & 0b00011111
        self.value_changed = True
        
    def add_loop(self):
        self.write_data()
        # A 1-length extended pause
        self.data.append(0b10100000)
        self.data.append(0b00000001)

    def write_data(self):
        frames = self.waits // 735
        
        if self.value_changed:
            if frames <= 4:
                print(f"Rhythm: value changed to {self.value:05b}, then wait {frames} frames")
                b = 0
                b |= (frames - 1) << 5
                b |= self.value & 0b00011111
                self.data.append(b)
                frames = 0
            else:
                print(f"Rhythm: value changed to {self.value:05b}")
                b = 0b10000000
                b |= self.value & 0b00011111
                self.data.append(b)
            self.value_changed = False
            
        if frames > 0:
            print(f"Rhythm: Pause {frames} frames")
            self.waits -= frames * 735
            while frames > 0:
                b = 0b10100000
                if frames <= 31:
                    # 1-byte count
                    b |= frames
                    self.data.append(b)
                    frames = 0
                elif frames <= 255 + 30: # 1 reserved length value
                    # 2-byte count
                    self.data.append(b)
                    self.data.append(frames - 30)
                    frames = 0
                else:
                    # 2-byte max count and loop
                    self.data.append(b)
                    self.data.append(255)
                    frames -= 255 + 30

    def terminate(self):
        # A zero-length extended pause
        self.data.append(0b10100000)
        self.data.append(0b00000000)

# FMLib format speculation:
#
# Each tone channel has its own sequence of pauses, frequency changes and key events.
# We also want to have a rhythm channel and another channel, for data not captured here?
# Is this better or worse than encoding it all in one stream?
# YM2413 registers cover:
# 1x: F-Num low bits (8)
# 2x: F-Num high bit (1); block (2); key (1); sustain (1)
# 3x: volume (4); instrument (4)
# * The note frequency is determined by F-Num * 2^block. This is likely to change very often.
#   It may be altered by vibrato, which leads to a lot of small changes (typically +/- 1 F-Num).
#   Block changes are less frequent.
#   If we represent the note frequency as F-Num * 2^block then it is a 14-bit number.
# * Key changes are of course the start and end of every note.
#   These are more likely to be next to large note changes.
# * Sustain seems likely to be changed less often than key
# * Volume and instrument changes are rare
# * Custom instrument may not be used at all, or if it is it's unlikely to change much
# We therefore try to optimise for data as so:
# 1. Small frequency changes with small waits
#    %00sxxyyy = change F-num by x+1, s = sign (1 -> -x, 0 -> +x) and wait y+1 frames (max 8)
# 2. Waits
#    %010xxxxx = wait x frames, max 31
#    %01000000 xxxxxxxx = wait x+32 frames, max 287
# 3. Note changes or key change
#    %101sbbbf ffffffff = block b, F-num f, implicit key on, sustain s
#    %10000000 = key off, no change to anything else?
#    %10000001 = end of stream
#    %10000010 = custom instrument data, next 8 bytes to registers 0..7
#    %10000011 = loop marker
#    %100----- = reserved except for above
# 4. Instrument or volume change
#    %011txxxx = instrument (t=1) or volume (t=0) change to x
# 5. Compression
#    %11xxxxxx = compressed data, see below
#
# Or, in bit order:
# %00sxxyyy => F-num delta plus wait
# %010xxxxx => wait, may have an extra byte if x=0
# %011txxxx => instrument (t=0) or volume (t=1) change
# %100xxxxx => special event
#              x = 0 => key up
#              x = 1 => end of stream
# %101sbbbf => start of key down, +1 extra byte for full note info
# %11xxxxxx => compression run of length x+4, followed by two bytes offset of run start relative to start of all data
#
# For rhythm data:
# 1. Rhythm keys down
#    %0nnkkkkk = write value k, rhythm enabled, pause n+1 frames (max 4)
#    %100kkkkk = write value k, rhythm enabled
# 2. Waits
#    %101nnnnn = pause n frames, max 31
#    %10100000 xxxxxxxx = pause x+30 frames, x>0, max 285 <- note different to tone channels
# 3. End of stream
#    %10100000 00000000 = end of stream
#    %10100000 00000001 = loop point
# 4. Compression
#    %11xxxxxx = compressed data
#
# Compressed data consists of:
# %11xxxxxxx oooooooo oooooooo
# This means that x+4 bytes (6-bit, max 259) from offset o (little-endian 16-bit)
# should be consumed next. Such runs cannot "nest".

# TODO support channel masking? Rhythm setup confuses this
# TODO compression!
# TODO looping - needs to not be in a "compressed" run? How to keep track of it? Must also break up any pauses. Maybe I need a special command for this.
# TODO allow dropping custom instrument data

def convert(filename):
    file = VgmFile(filename)
    print(f"Loaded file, size is {len(file.data)}")
    print(f"Data starts at {file.data_offset:#x}")
    print(f"Data loops at {file.loop_offset:#x}")
    print(f"Data ends at {file.eof_offset:#x}")

    channels = {}
    for index in range(9):
        channels[index] = ToneChannel(index)
    
    channels[9] = RhythmChannel()
    
    # Parse the file into memory
    commands = [command for command in file.commands()]

    first_key_channel = 0
    for line in commands:
        if type(line) is Command:
            if line.register >= 0x20 and line.register <= 0x28 and line.data & 0b00010000 != 0:
                # It's a key press
                first_key_channel = line.register & 0xf
                print(f"First key press is {line} for channel {first_key_channel}")
                break

    for line in commands:
        print(line)
        if type(line) is Wait:
            for i in channels:
                channels[i].wait(line.length)
        elif type(line) is LoopPoint:
            print("loop point")
            for i in channels:
                channels[i].add_loop()
        elif type(line) is Command:
            if line.register >= 0x10:
                # tone register
                channel = line.register & 0xf
                if channel > 8:
                    continue # shouldn't happen
                if line.register & 0xf0 == 0x10:
                    channels[channel].reg1x(line.data)
                elif line.register & 0xf0 == 0x20:
                    channels[channel].reg2x(line.data)
                else:
                    channels[channel].reg3x(line.data)
            elif line.register == 0x0e:
                # rhythm
                channels[9].write(line.data)
            elif line.register <= 0x08:
                # custom instrument, we load that onto the channel with the first key down
                channels[first_key_channel].add_custom(line.register, line.data)
            else:
                # Something else
                print(f"Unhandled command for register {line.register}")
                pass
                
    # Terminate
    for index in channels:
        # Write any remaining waits
        channels[index].write_data()
        channels[index].terminate()

    # Print some stuff
    for index in channels:
        print(f"ch{index} data = {len(channels[index].data)} bytes")
        
    # Dump data out, raw mess for now
    with open(filename + ".fm", "wb") as f:
        for index in channels:
            f.write(channels[index].data)

    
def main():
    verb = sys.argv[1]
    if verb == 'convert':
        convert(sys.argv[2])
    else:
        raise Exception(f"Unknown verb \"{verb}\"")


if __name__ == "__main__":
    main()
