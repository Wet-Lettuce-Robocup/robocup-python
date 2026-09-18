#!/usr/bin/env python3
import smbus

bus = smbus.SMBus(1)
bus.write_byte(0x67, 0x02)
bus.close()
