#!/usr/bin/env python3

from switchyard.lib.address import *
from switchyard.lib.packet import *
from switchyard.lib.userlib import *
from threading import *
import random
import time

import os

# hard code
# middle_eth1 = 
# blastee_eth = 
# middle_eth0 =
# blaster_eth = 

SEED = None
DROP_P = None

def drop(percent):
    return random.randrange(100) < percent

class MiddleBox(object):
    def __init__(self, net, params_file):
        self.net = net
        self.parse_params(params_file)

    def parse_params(self, params_file):
        with open(params_file, 'r') as fp:
            params = fp.readline().strip().split(' ')
            i = 0
            while i < len(params):
                if params[i] == "-s":
                    SEED = int(params[i+1]);    i += 2
                elif params[i] == "-p":
                    DROP_P = int(params[i+1]);  i += 2
                else:
                    i += 1
            if not SEED or not DROP_P:
                print ("Error: not find seed or drop_p")
                os._exit(1)
            print ("seed = " + str(SEED) + ", drop_p = " + str(DROP_P))

        def run_middlebox(self):
            my_intf = self.net.interfaces()
            mymacs = [intf.ethaddr for intf in my_intf]
            myips = [intf.ipaddr for intf in my_intf]

            # random.seed(random_seed) #Extract random seed from params file
            random.seed(SEED)

            while True:
                gotpkt = True
                try:
                    timestamp, dev, pkt = self.net.recv_packet()
                    log_debug("Device is {}".format(dev))

                    if dev == "middlebox-eth0":  # Receive pkt from blaster
                        if not drop(self.drop_p):
                            # modify headers & send to blastee
                            pkt.get_header(Ethernet).src = middle_eth1
                            pkt.get_header(Ethernet).dst = blastee_eth

                            self.net.send_packet("middlebox-eth1", pkt)
                    elif dev == "middlebox-eth1":       # Receive ACK from blastee
                        # Modify headers & send to blaster. Not dropping ACK packets!
                        pkt.get_header(Ethernet).src = middle_eth0
                        pkt.get_header(Ethernet).dst = blaster_eth

                        self.net.send_packet("middlebox-eth0", pkt)
                    else:
                        log_debug("Oops :(")
                except NoPackets:
                    log_debug("No packets available in recv_packet")
                    gotpkt = False
                except Shutdown:
                    log_debug("Got shutdown signal")
                    break

                if gotpkt:
                    log_debug("I got a packet {}".format(pkt))

                
def main(net):
    middlebox = MiddleBox(net, params_file="middlebox_params.txt")
    middlebox.run_middlebox()
    net.shutdown()

# test
# middlebox = MiddleBox(None, "middlebox_params.txt")