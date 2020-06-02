#!/usr/bin/env python3

from switchyard.lib.address import *
from switchyard.lib.packet import *
from switchyard.lib.userlib import *
from random import randint
import time

import os

# hard code
# blastee_eth = 
# blastee_ip = 
# blaster_eth = 
# blaster_ip = 

MAX_SEQNUM = None
PAYLOAD_LENGTH = None
WINDOW = None
TIMEOUT = None
RECV_TIMEOUT = None

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
        self.parse_params(params_file)
        self.seqnum2pktinfo = dict()

    def parse_params(self, params_file):
        with open(params_file, 'r') as fp:
            params = fp.readline().strip().split(' ')
            i = 0
            while i < len(params):
                # -n 15 -l 16 -w 5 -t 1000 -r 150
                if params[i] == "-n":
                    MAX_SEQNUM = int(params[i+1]);     i += 2
                elif params[i] == "-l":
                    PAYLOAD_LENGTH = int(params[i+1]);  i += 2
                elif params[i] == "-w":
                    WINDOW = int(params[i+1]);          i += 2
                elif params[i] == "-t":
                    TIMEOUT = float(params[i+1])/ 1000;         i += 2
                elif params[i] == "-r":
                    RECV_TIMEOUT = float(params[i+1])/ 1000;    i += 2
                else:
                    i += 1
            if not MAX_SEQNUM or not PAYLOAD_LENGTH or not WINDOW or not TIMEOUT or not RECV_TIMEOUT:
                print ("Error: do not acquire some needed parameters")
                os._exit(1)
            print (str(MAX_SEQNUM) + ' ' + str(PAYLOAD_LENGTH) + ' ' + str(WINDOW) + ' ' + str(TIMEOUT) + ' ' + str(RECV_TIMEOUT) + ' ')

    def construct_pkt(self, seqnum):
        # ToDo
        return pkt

    def get_seqnum_from_ack(self, pkt):
        # ToDo
        return seqnum

    def run_blaster(self):
        my_intf = self.net.interfaces()
        mymacs = [intf.ethaddr for intf in my_intf]
        myips = [intf.ipaddr for intf in my_intf]

        left, right = 1, 1

        while True:
            gotpkt = True
            try:
                #Timeout value will be parameterized!
                timestamp, dev, ack = self.net.recv_packet(RECV_TIMEOUT)
            except NoPackets:
                log_debug("No packets available in recv_packet")
                gotpkt = False
            except Shutdown:
                log_debug("Got shutdown signal")
                break

            if left == 1 and right == 1:
                pkt = self.construct_pkt(right)
                self.seqnum2pktinfo[right] = PktInfo(pkt, time.time())
                self.net.send_packet(my_interfaces[0].name, pkt)
                if right < MAX_SEQNUM:
                    right += 1
                continue
                
            if gotpkt:
                seqnum = self.get_seqnum_from_ack(ack)
                self.seqnum2pktinfo[seqnum].acked = True

                if seqnum == left:
                    # set left to next un-acked seqnum
                    while self.seqnum2pktinfo[seqnum].acked:
                        left += 1
                if left > right:
                    break
            
            resend = False
            for it_resend in range(left, right):
                if not self.seqnum2pktinfo[it_resend].acked and \
                    time.time() - self.seqnum2pktinfo[it_resend].timestamp > TIMEOUT:
                    self.seqnum2pktinfo[it_resend].timestamp = time.time()
                    self.net.send_packet(my_interfaces[0].name, self.seqnum2pktinfo[it_resend].pkt)
                    resend = True
                    break
            if resend:
                continue    
            
            # send new pkt
            if right - left < WINDOW and right < MAX_SEQNUM:
                pkt = self.construct_pkt(right)
                self.seqnum2pktinfo[right] = PktInfo(pkt, time.time())
                self.net.send_packet(my_interfaces[0].name, pkt)
                right += 1

            
                


def main(net):
    blaster = Blaster(net, 'blaster_params.txt')
    blaster.run_blaster()
    # print 
    net.shutdown()

# test
blaster = Blaster(None, 'blaster_params.txt')