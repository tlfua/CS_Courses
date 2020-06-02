#!/usr/bin/env python3

from switchyard.lib.address import *
from switchyard.lib.packet import *
from switchyard.lib.userlib import *
from random import randint
import time

import os

# hard code
blaster_eth = '10:00:00:00:00:01'
middle_eth0 = '40:00:00:00:00:01'

blastee_ip = '192.168.200.2'
blaster_ip = '192.168.100.2'


def print_output(total_time, num_ret, num_tos, throughput, goodput):
    print("Total TX time (s): " + str(total_time))
    print("Number of reTX: " + str(num_ret))
    print("Number of coarse TOs: " + str(num_tos))
    print("Throughput (Bps): " + str(throughput))
    print("Goodput (Bps): " + str(goodput))


class PktInfo(object):
    def __init__(self, pkt, timestamp):
        self.pkt = pkt
        self.acked = False
        self.timestamp = timestamp


class Blaster(object):
    def __init__(self, net, params_file):
        self.net = net
        # changed

        self.MAX_SEQNUM = -1
        self.PAYLOAD_LENGTH = -1
        self.WINDOW = -1
        self.TIMEOUT = -1
        self.RECV_TIMEOUT = -1

        self.parse_params(params_file)
        self.seqnum2pktinfo = dict()
        self.total_sent_pkts = 0

    def parse_params(self, params_file):
        with open(params_file, 'r') as fp:
            params = fp.readline().strip().split(' ')
            i = 0
            while i < len(params):
                # -n 15 -l 16 -w 5 -t 1000 -r 150
                if params[i] == "-n":
                    self.MAX_SEQNUM = int(params[i + 1]);
                    i += 2
                elif params[i] == "-l":
                    self.PAYLOAD_LENGTH = int(params[i + 1]);
                    i += 2
                elif params[i] == "-w":
                    self.WINDOW = int(params[i + 1]);
                    i += 2
                elif params[i] == "-t":
                    self.TIMEOUT = float(params[i + 1]) / 1000;
                    i += 2
                elif params[i] == "-r":
                    self.RECV_TIMEOUT = float(params[i + 1]) / 1000;
                    i += 2
                else:
                    i += 1
            if self.MAX_SEQNUM < 0 or self.PAYLOAD_LENGTH < 0 or self.WINDOW < 0 or self.TIMEOUT < 0 or self.RECV_TIMEOUT < 0:
                print ("Error: do not acquire some needed parameters")
                os._exit(1)
            print(str(self.MAX_SEQNUM) + ' ' + str(self.PAYLOAD_LENGTH) + ' ' + str(self.WINDOW) + ' ' + str(
                self.TIMEOUT) + ' ' + str(self.RECV_TIMEOUT) + ' ')

    def construct_pkt(self, seqnum):
        # ToDo
        eth = Ethernet()
        eth.src = blaster_eth
        eth.dst = middle_eth0

        ip = IPv4(protocol=IPProtocol.UDP)
        ip.src = blaster_ip
        ip.dst = blastee_ip

        udp = UDP()

        pkt = eth + ip + udp
        pkt += seqnum.to_bytes(4, 'big')
        pkt += self.PAYLOAD_LENGTH.to_bytes(2, 'big')
        pkt += os.urandom(self.PAYLOAD_LENGTH)
        return pkt

    def get_seqnum_from_ack(self, pkt):
        # ToDo
        contents = pkt.get_header(RawPacketContents)
        seqnum = int.from_bytes(contents.data[:4], 'big')
        return seqnum

    def run_blaster(self):
        my_interfaces = self.net.interfaces()
        # changed
        num_ret = 0
        num_tos = 0

        left, right = 1, 1

        while True:
            gotpkt = True
            try:
                # Timeout is parameterized!
                timestamp, dev, ack = self.net.recv_packet(timeout=self.RECV_TIMEOUT)
            except NoPackets:
                log_debug("No packets available in recv_packet")
                gotpkt = False
            except Shutdown:
                log_debug("Got shutdown signal")
                break

            # 5/28 modified
            # print every seqnum is acked or not
            print("left = " + str(left) + ", right = " + str(right))
            for i in range(1, self.MAX_SEQNUM + 1):
                if i in self.seqnum2pktinfo.keys():
                    print (i, self.seqnum2pktinfo[i].acked, end=" ")
            print ()
            ########


            if left == 1 and right == 1:
                pkt = self.construct_pkt(right)
                self.seqnum2pktinfo[right] = PktInfo(pkt, time.time())
                # changed

                self.firstpkt_send_time = time.time()

                self.net.send_packet(my_interfaces[0].name, pkt)
                # changed
                self.total_sent_pkts += 1

                if right < self.MAX_SEQNUM + 1:
                    right += 1

                continue

            if gotpkt:
                seqnum = self.get_seqnum_from_ack(ack)
                print("sequnum is  " + str(seqnum))
                # changed
                if seqnum in self.seqnum2pktinfo:
                    self.seqnum2pktinfo[seqnum].acked = True
                # set left to next un-acked seqnum
                if seqnum == left:
                    # 5/28 modified
                    # fix 'self.seqnum2pktinfo[seqnum].acked' typo
                    while left < right and self.seqnum2pktinfo[left].acked:
                        left += 1
                # 5/28 modified
                # when all seqnum are acked, it satisfy below condition to break
                if left == right and right == self.MAX_SEQNUM + 1:
                    self.last_ack_time = time.time()
                    break

            resend = False
            for it_resend in range(left, right):
                if not self.seqnum2pktinfo[it_resend].acked and \
                        time.time() - self.seqnum2pktinfo[it_resend].timestamp > self.TIMEOUT:
                    print ("resend " + str(it_resend)) # check
                    self.seqnum2pktinfo[it_resend].timestamp = time.time()
                    self.net.send_packet(my_interfaces[0].name, self.seqnum2pktinfo[it_resend].pkt)

                    # changed
                    num_ret += 1
                    num_tos += 1

                    resend = True
                    break
            if resend:
                continue

            # send new pkt
            # 5/28 modified
            # modify the final position the right pointer could be
            if right - left < self.WINDOW and right < self.MAX_SEQNUM + 1:
                pkt = self.construct_pkt(right)
                self.seqnum2pktinfo[right] = PktInfo(pkt, time.time())
                self.net.send_packet(my_interfaces[0].name, pkt)

                # changed
                self.total_sent_pkts += 1

                right += 1

        # stats
        total_time = self.last_ack_time - self.firstpkt_send_time

        throughput = (self.total_sent_pkts * self.PAYLOAD_LENGTH) / total_time

        goodput = (self.MAX_SEQNUM * self.PAYLOAD_LENGTH) / total_time

        return total_time, num_ret, num_tos, throughput, goodput


def main(net):
    blaster = Blaster(net, 'blaster_params.txt')
    total_time, num_ret, num_tos, throughput, goodput = blaster.run_blaster()
    # print
    print("Finish!!!!!!!")
    print_output(total_time, num_ret, num_tos, throughput, goodput)
    net.shutdown()

# test
# blaster = Blaster(None, 'blaster_params.txt')
