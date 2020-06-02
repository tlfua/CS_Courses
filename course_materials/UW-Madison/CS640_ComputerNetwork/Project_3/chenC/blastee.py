#!/usr/bin/env python3

from switchyard.lib.address import *
from switchyard.lib.packet import *
from switchyard.lib.userlib import *
from threading import *
import time

import os
from base64 import b64encode


# hard code
blastee_eth = '20:00:00:00:00:01'
middle_eth1 = '40:00:00:00:00:02'

blastee_ip = '192.168.200.2'
blaster_ip = '192.168.100.2'

class Blastee(object):
    def __init__(self, net):
        self.net = net

    def construct_ack(self, pkt):
        # pkt_eth_src = pkt.get_header(Ethernet).src
        # pkt_eth_dst = pkt.get_header(Ethernet).dst
        # pkt_ip_src = pkt.get_header(IPv4).src
        # pkt_ip_dst = pkt.get_header(IPv4).dst

        contents = pkt.get_header(RawPacketContents)
        # seq_num = int.from_bytes(contents.data[:4], 'big')
        seq_num_bytes = contents.data[:4]

        payload_bytes = None
        if len(contents.data[4:]) >= 8:
            payload_bytes = contents.data[4:12]
        else:
            payload_bytes = contents.data[4:]

        eth = Ethernet()
        eth.src = blastee_eth
        eth.dst = middle_eth1

        ip = IPv4(protocol=IPProtocol.UDP)
        ip.src = blastee_ip
        ip.dst = blaster_ip

        udp = UDP()

        ack = eth + ip + udp + seq_num_bytes + payload_bytes
        return ack


    def run_blastee(self):
        my_interfaces = self.net.interfaces()
        # mymacs = [intf.ethaddr for intf in my_interfaces]
        # Is it supposed to only contain one interface?
        if len(my_interfaces) != 1:
            print ("More than one interface")
            os._exit(1)

        while True:
            gotpkt = True
            try:
                timestamp, dev, pkt = self.net.recv_packet()
                log_debug("Device is {}".format(dev))

                ack = self.construct_ack(pkt)
                self.net.send_packet(my_interfaces[0].name, ack)
            except NoPackets:
                log_debug("No packets available in recv_packet")
                gotpkt = False
            except Shutdown:
                log_debug("Got shutdown signal")
                break

            if gotpkt:
                log_debug("I got a packet from {}".format(dev))
                log_debug("Pkt: {}".format(pkt))

def main(net):
    blastee = Blastee(net)
    blastee.run_blastee()
    net.shutdown()