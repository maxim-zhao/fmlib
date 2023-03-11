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


# TODO nest this inside ToneChannel?
class ToneState:
    def __init__(self):
        self.fnum = 0
        self.block = 0
        self.key = False
        self.sustain = False
        self.volume = 0
        self.instrument = 0


class ToneChannel(Channel):    
    def __init__(self, index):
        super().__init__()
        self.channel = index
        self.state = ToneState()
        self.last_state = ToneState()
        self.raw = bytearray()

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

    def write_data(self):
        # We want to encode some data for what's changed since last time, then the pause (if non-zero)
        frames = self.waits // 735
        delta = self.state.fnum - self.last_state.fnum
        # If key has gone from down to up, we handle that first
        if (not self.state.key) and self.last_state.key:
            print(f"ch{self.channel}: Key up")
            self.data.append(0b01000000)
            self.last_state.key = self.state.key
            # And we ignore any other changes... this is not quite VGM compatible
            delta = 0
        else:
            # Check for a large f-num change or other change affecting block, key(, sustain)
            if delta < -4 or delta > 4 or self.state.block != self.last_state.block or self.state.sustain != self.last_state.sustain or self.state.key != self.last_state.key:
                print(f"ch{self.channel}: Changed from key {self.last_state.key}, sustain {self.last_state.sustain}, block {self.last_state.block}, f-num {self.last_state.fnum} to key {self.state.key}, sustain {self.state.sustain}, block {self.state.block}, f-num {self.state.fnum}")
                b = 0b01100000
                if self.state.key:
                    b |= 0b00010000
                #if self.sustain: TODO where does sustain go?!
                #    b |= 0b00001000
                b |= self.state.fnum >> 8
                b |= self.state.block << 1
                self.data.append(b)
                self.data.append(self.state.fnum & 0xff)
                # Remember the changes
                self.last_state.key = self.state.key
                #self.last_state.sustain = self.state.sustain TODO here too
                self.last_state.block = self.state.block
                self.last_state.fnum = self.state.fnum
                # This captures 3 or 6 bytes of VGM to 2 bytes
            
            if self.state.volume != self.last_state.volume:
                print(f"ch{self.channel}: volume changed to {self.state.volume}")
            if self.state.instrument != self.last_state.instrument:
                print(f"ch{self.channel}: instrument changed to {self.state.instrument}")
            if self.state.volume != self.last_state.volume or self.state.instrument != self.last_state.instrument:
                self.raw.append(0x30 + self.channel)
                self.raw.append(self.state.instrument << 4 | self.state.volume)
                self.last_state.volume = self.state.volume
                self.last_state.instrument = self.state.instrument
                # This captures 3 bytes of VGM to 3 bytes :( but it's pretty rare..?
            
        while len(self.raw) > 0:
            count = len(self.raw) // 2 # Cannot be more than 32?
            print(f"ch{self.channel}: Emitting {count} raw data pairs")
            self.data.append(0b10000000 | count)
            self.data += self.raw
            self.raw = bytearray()
            
        if self.state.fnum != self.last_state.fnum:
            # Must be a small change left over from above, not accompanied by anything else important
            print(f"ch{self.channel}: Changed f-num from {self.last_state.fnum} to {self.state.fnum} ({delta}) and wait is {frames} frames")
            b = 0
            frames_to_consume = min(frames, 4)
            if delta < 0:
                b |= 0b00010000
                delta *= -1
            b |= delta << 2
            b |= frames_to_consume # Consume up to 4 frames
            self.data.append(b)
            self.waits -= frames_to_consume * 735
            # This captures 4 bytes of VGM to 1 byte

        if self.waits >= 735:
            frames = self.waits // 735
            print(f"ch{self.channel}: Pause {frames} frames")
            self.waits -= frames * 735
            while frames > 0:
                b = 0b11000000
                if frames <= 32:
                    b |= frames + 1
                    self.data.append(b)
                    frames = 0
                elif frames <= 256:
                    self.data.append(b)
                    self.data.append(frames - 1)
                    frames = 0
                else:
                    self.data.append(b)
                    self.data.append(255)
                    frames -= 256


class RhythmChannel(Channel):
    def __init__(self):
        super().__init__()
        self.value = 0
        self.value_changed = False
        
    def write(self, data):
        self.maybe_write_data()
        self.value = data
        self.value_changed = True
        
    def write_data(self):
        if self.value_changed:
            print(f"Rhythm: value changed to {self.value:b}")
            self.data.append(self.value & 0b00111111)
            self.value_changed = False
            
        if self.waits >= 735:
            frames = self.waits // 735
            print(f"Rhythm: Pause {frames} frames")
            self.waits -= frames * 735
            while frames > 0:
                b = 0b11000000
                if frames <= 32:
                    b |= frames + 1
                    frames = 0
                else:
                    b |= 31
                    frames -= 32
                self.data.append(b)


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
# We therefore try to optimise for data as so:
# 1. Small frequency changes (vibrato, +/- <16?)
#    %000xxxyy = change F-num by x (1's-comp signed, -4..+4) and wait y+1 frames (max 4)
# 2. Waits
#    %001xxxxx = wait x+1 frames, max 32
#    %00100000 xxxxxxxx = wait x+1 frames, max 256
# 3. Note changes + key
#    %0100---- = key off
#    %0101bbbx xxxxxxxx = block b, F-num x, key on
# 4. Raw data
#    %011nnnnn = n+1 register, data pairs - i.e. max 32 pairs, 64 bytes
# For rhythm data, we can have a simpler format:
# 1. Rhythm keys
#    %00kkkkkk = value to write to rhythm control register
# 2. Waits
#    Same as above, or use extra bits?
# TODO what about sustain?
# TODO what about custom instruments? -> raw data on a tone channel? But which one?
# TODO support channel masking? Rhythm setup confuses this

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

    for line in file.commands():
        print(line)
        if type(line) is Wait:
            for i in channels:
                channels[i].wait(line.length)
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
            else:
                # Something else
                # TODO
                pass
                
    # Print some stuff
    for index in channels:
        print(f"ch{index} data = {len(channels[index].data)} bytes")

def main():
    verb = sys.argv[1]
    if verb == 'convert':
        convert(sys.argv[2])
    else:
        raise Exception(f"Unknown verb \"{verb}\"")


if __name__ == "__main__":
    main()
