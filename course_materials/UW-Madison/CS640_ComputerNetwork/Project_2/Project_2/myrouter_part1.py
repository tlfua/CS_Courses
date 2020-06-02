#!/usr/bin/env python3

'''
Basic IPv4 router (static routing) in Python.
'''

import sys
import os
import time

from switchyard.lib.packet.util import *
from switchyard.lib.userlib import *

class Router(object):
    def __init__(self, net):
        self.net = net
        # other initialization stuff here
        self.arpTable , self.IP_list = self.Build_arpTable_IPlist()
        
    def Build_arpTable_IPlist(self):
        arpTable = {}
        ip_list = []
        for intf in self.net.interfaces():
           arpTable.update({intf.ipaddr : intf.ethaddr})
           ip_list.append(intf.ipaddr)          
        return arpTable, ip_list

    def arpreply_pkt(self,senderhwaddr, targethwaddr, senderprotoaddr,targetprotoaddr):
        pkt = create_ip_arp_reply(senderhwaddr, targethwaddr, senderprotoaddr,targetprotoaddr)
        return pkt
        

 
    def router_main(self):    
        '''
        Main method for router; we stay in a loop in this method, receiving
        packets until the end of time.
        '''
        while True:
            gotpkt = True
            try:
                timestamp,dev,pkt = self.net.recv_packet(timeout=1.0)
                eth = pkt.get_header(Ethernet)
                etherType = eth.ethertype
            
                if etherType == EtherType.ARP:
                    arp = pkt.get_header(Arp) 
                    operation = arp.operation
                    #source IP
                    senderprotoaddr = arp.senderprotoaddr
                    #source ether
                    senderhwaddr = arp.senderhwaddr
                    #dst IP
                    targetprotoaddr = arp.targetprotoaddr
                    if operation == ArpOperation.Request:
                        if targetprotoaddr in self.IP_list:
                        #get dst ether
                            targethwaddr = self.arpTable[targetprotoaddr]
                            reply_pkt = self.arpreply_pkt(targethwaddr,senderhwaddr,targetprotoaddr,senderprotoaddr)
                            self.net.send_packet(dev,reply_pkt)
                    elif operation == ArpOperation.Reply:
                        if targetprotoaddr in self.IP_list:
                            self.arpTable.update({senderprotoaddr : senderhwaddr})

            except NoPackets:
                log_debug("No packets available in recv_packet")
                gotpkt = False
            except Shutdown:
                log_debug("Got shutdown signal")
                break

 
            if gotpkt:
                log_debug("Got a packet: {}".format(str(pkt)))



def main(net):
    '''
    Main entry point for router.  Just create Router
    object and get it going.
    '''
    r = Router(net)
    r.router_main()
    net.shutdown()
