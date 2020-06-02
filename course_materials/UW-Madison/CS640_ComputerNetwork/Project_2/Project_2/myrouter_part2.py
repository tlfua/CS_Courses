#!/usr/bin/env python3

'''
  Basic IPv4 router for the Computer Networks course CS640
'''

import sys
import os
import time

from switchyard.lib.packet.util import *
from switchyard.lib.userlib import *

MAX_ARP_REQS = 3

class WaitItem(object):
    def __init__(self, ip_pkt, next_hop_ip, next_hop_dev, arprequest):
        self.ip_pkt = ip_pkt
        self.next_hop_ip = next_hop_ip
        self.next_hop_dev = next_hop_dev
        self.arprequest = arprequest
        self.arpreq_sent_count = 1
        self.timestamp = time.time()
        self.to_delete = False

class Router(object):
    def __init__(self, net):
        # Initializing the router
        log_info("Initializing router.. ")


        self.net = net
        # self.forwarding_table = Router.parse_forwaring_table(static_table)
        self.forwarding_table = self.parse_and_update_forwaring_table()
        for k, v in self.forwarding_table.items():
            print (k, v)
        print ("fwd_table len = " + str(len(self.forwarding_table)))

        self.ips = self.get_intf_ips()
        # print (self.ips)
        self.arp_table = self.construct_arp_table()
        
        self.wait_queue = []

        for intf in self.net.interfaces():
            print (intf.ipaddr, intf.ethaddr, intf.name)

    def parse_and_update_forwaring_table(self):
        # parse
        forwarding_table = {}
        with open("forwarding_table.txt", 'r') as fp:
            data = fp.read().strip().split("\n")
            for line in data:
                net, subnet, next_hop, eth_port = line.strip().split()
                forwarding_table[IPAddr(net)] = (IPAddr(subnet), IPAddr(next_hop), eth_port)
        # update
        for intf in self.net.interfaces():
            net = IPv4Network(int(IPv4Address(intf.ipaddr)) & int(IPv4Address(intf.netmask)))
            forwarding_table[net.network_address] = (intf.netmask, intf.ipaddr, intf.name)
        return forwarding_table

    @staticmethod
    def make_arp_request(senderhwaddr, senderprotoaddr, targetprotoaddr):
        # Creates the ARP request
        ether = Ethernet()
        ether.src = senderhwaddr
        ether.dst = 'ff:ff:ff:ff:ff:ff'
        ether.ethertype = EtherType.ARP
        arp = Arp(operation=ArpOperation.Request,\
                  senderhwaddr=senderhwaddr,\
                  senderprotoaddr=senderprotoaddr,\
                  targethwaddr='ff:ff:ff:ff:ff:ff',\
                  targetprotoaddr=targetprotoaddr)
        arppacket = ether + arp
        return arppacket

    def get_intf_ips(self):
        return [intf.ipaddr for intf in self.net.interfaces()]

    def get_intf(self, ethaddr):
        for intf in self.net.interfaces():
            if intf.ethaddr == ethaddr:
                return intf

    def construct_arp_table(self):
        return {intf.ipaddr: intf.ethaddr for intf in self.net.interfaces()}

    def make_arp_reply(self, dev, senderhwaddr, targethwaddr, senderprotoaddr,
                       targetprotoaddr):
        pkt = create_ip_arp_reply(senderhwaddr, targethwaddr, senderprotoaddr,
                                  targetprotoaddr)
        self.net.send_packet(dev, pkt)

    def find_next_hop(self, dest_addr):
        dst_addr = IPv4Address(dest_addr)

        max_prefixlen = 0
        ret_next_hop_addr, ret_next_hop_dev = None, None
        for net_addr, value in self.forwarding_table.items():
            mask, next_hop_addr, next_hop_dev = value

            if (int(net_addr) & int(dst_addr)) == int(net_addr):
                prefixlen = IPv4Network(str(net_addr) + '/' + str(mask)).prefixlen
                if prefixlen > max_prefixlen:
                    max_prefixlen = prefixlen
                    ret_next_hop_addr = next_hop_addr
                    ret_next_hop_dev = next_hop_dev
        # if not ret_next_hop_addr or not ret_next_hop_dev:
        #     print ("Error: can not find next hop")
        #     os._exit(1)
        return ret_next_hop_addr, ret_next_hop_dev

    def send_ip(self, dev, next_hop_ip, ip_pkt):
        # intf = self.net.interface_by_name(dev)

        eth_payload = ip_pkt.get_header(Ethernet)
        # eth_payload.src = intf.ethaddr
        eth_payload.dst = self.arp_table[next_hop_ip]

        ip_payload = ip_pkt[IPv4]
        ip_payload.ttl -= 1
                
        log_info("Forwarding IP pkt: %s" % ip_pkt)
        self.net.send_packet(dev, ip_pkt)
        # else:
        #     arprequest = Router.make_arp_request(intf.ethaddr, intf.ipaddr, next_hop_ip)
        #     self.net.send_packet(dev, arprequest)

    def maintain_wait_queue(self, dev, next_hop_ip, ip_pkt):
        pass

    def router_main(self):    
        '''
        Main method for router; we stay in a loop in this method, receiving
        packets until the end of time.
        '''
        while True:
            gotpkt = True

            try:
                timestamp, dev, pkt = self.net.recv_packet(timeout=1.0)
                eth_payload = pkt.get_header(Ethernet)
                
                if eth_payload.ethertype == EtherType.ARP:  # ARP
                    arp_header = pkt.get_header(Arp)
                    if arp_header.operation == ArpOperation.Request:  # ARP Request
                        targetprotoaddr = arp_header.targetprotoaddr
                        if targetprotoaddr in self.ips:
                            targethwaddr = self.arp_table[targetprotoaddr]
                            self.make_arp_reply(dev, targethwaddr,
                                                arp_header.senderhwaddr,
                                                targetprotoaddr,
                                                arp_header.senderprotoaddr)

                    if arp_header.operation == ArpOperation.Reply:  # ARP Reply
                        ipaddr = arp_header.senderprotoaddr
                        ethaddr = arp_header.senderhwaddr
                        self.arp_table[ipaddr] = ethaddr # update ARP table
                elif eth_payload.ethertype == EtherType.IP:  # IP
                    ip_payload = pkt.get_header(IPv4)
                    dst_ip = ip_payload.dst
                    next_hop_ip, next_hop_dev = self.find_next_hop(dst_ip)

                    if dst_ip not in self.ips and (next_hop_ip and next_hop_dev):
                        # meaning the next hop is dst
                        if next_hop_ip in self.ips:
                            ip_payload = pkt.get_header(IPv4)
                            next_hop_ip = ip_payload.dst

                        if next_hop_ip in self.arp_table.keys():
                            # just send ip
                            self.send_ip(next_hop_dev, next_hop_ip, pkt)
                        else:
                            # maintain wait_queue
                            intf = self.net.interface_by_name(next_hop_dev)
                            arprequest = Router.make_arp_request(intf.ethaddr, intf.ipaddr, next_hop_ip)
                            self.net.send_packet(next_hop_dev, arprequest) # send arp req the first time
                            wait_item = WaitItem(pkt, next_hop_ip, next_hop_dev, arprequest)
                            self.wait_queue += [wait_item]

                # check if wait_item get the proper arp reply to send ip pkt
                # check if next_hop_ip get the proper mac addr
                for wait_item in self.wait_queue:
                    if wait_item.next_hop_ip in self.arp_table.keys():
                        self.send_ip(wait_item.next_hop_dev, wait_item.next_hop_ip, wait_item.ip_pkt)
                        wait_item.to_delete = True
                    else:
                        cur_time = time.time()
                        if cur_time - wait_item.timestamp > 1:
                            if wait_item.arpreq_sent_count < MAX_ARP_REQS:
                                wait_item.timestamp = cur_time
                                self.net.send_packet(wait_item.next_hop_dev, wait_item.arprequest)
                                wait_item.arpreq_sent_count += 1
                            else:
                                wait_item.to_delete = True
                tmp = []
                for wait_item in self.wait_queue:
                    if wait_item.to_delete == False:
                        tmp += [wait_item]
                self.wait_queue = tmp

            except NoPackets:
                log_debug("No packets available in recv_packet")
                gotpkt = False

            except Shutdown:
                log_debug("Got shutdown signal")
                break

            except NoMatchFoundException as e:
                self.icmp_error_handler(
                    pkt, ICMPType.DestinationUnreachable,
                    ICMPCodeDestinationUnreachable.NetworkUnreachable, str(e)
                )

            except TTLExpiredException as e:
                self.icmp_error_handler(
                    pkt, ICMPType.TimeExceeded, ICMPCodeTimeExceeded.TTLExpired,
                    str(e)
                )

            except DestinationPortUnreachableException as e:
                self.icmp_error_handler(
                    pkt, ICMPType.DestinationUnreachable,
                    ICMPCodeDestinationUnreachable.PortUnreachable, str(e)
                )

            if gotpkt:
                log_debug("Got a packet: {}".format(str(pkt)))


class NoMatchFoundException(Exception):
    pass

class TTLExpiredException(Exception):
    pass

class DestinationPortUnreachableException(Exception):
    pass


def main(net):
    '''
    Main entry point for router.  Just create Router
    object and get it going.
    '''
    print ("myswitch_part2: 3.27 new version")

    r = Router(net)
    r.router_main()
    net.shutdown()
